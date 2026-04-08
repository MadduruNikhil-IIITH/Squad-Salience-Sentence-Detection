"""
RST (Rhetorical Structure Theory) Parser Wrapper

Parses passages into discourse trees using isanlp_rst.
Extracts sentence-level and intra-sentence discourse features.

Key Features:
- Lazy initialization of RST parser (heavy model load on first call)
- Passage-level caching to avoid redundant parses
- Sentence-to-EDU mapping for feature extraction
- Intra-sentence clause detection
- Error handling and edge case management
"""

import hashlib
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

try:
    from isanlp_rst.parser import Parser
except ImportError:
    raise ImportError(
        "isanlp_rst not found. Install with: "
        "pip install git+https://github.com/iinemo/isanlp.git isanlp_rst"
    )

# ============================================================================
# Global Parser (lazy-loaded on first call)
# ============================================================================

_PARSER: Optional[Parser] = None
_PARSER_INITIALIZED = False


def _get_parser() -> Parser:
    """Initialize and return the RST parser (lazy-loaded on first use)."""
    global _PARSER, _PARSER_INITIALIZED

    if _PARSER is None:
        if not _PARSER_INITIALIZED:
            print("[RST Parser] Initializing... (first call, ~500MB model download)")
            _PARSER = Parser(
                hf_model_name='tchewik/isanlp_rst_v3',
                hf_model_version='gumrrg',  # Trained on GUM corpus (Wikipedia-like)
                cuda_device=0  # Use GPU if available; -1 for CPU
            )
            _PARSER_INITIALIZED = True
            print("[RST Parser] ✓ Initialized successfully")
        else:
            raise RuntimeError("Parser initialization attempted but failed")

    return _PARSER


# ============================================================================
# Caching Layer
# ============================================================================

class _RSTTreeCache:
    """LRU cache for parsed RST trees (key: passage hash, value: tree structure)."""

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._store: "OrderedDict[str, Dict]" = OrderedDict()

    def _hash_passage(self, text: str) -> str:
        """Compute stable hash for passage."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, passage: str) -> Optional[Dict]:
        """Retrieve cached tree if exists."""
        key = self._hash_passage(passage)
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)  # Mark as recently used
        return value

    def put(self, passage: str, value: Dict) -> None:
        """Store tree in cache."""
        key = self._hash_passage(passage)
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)  # Remove oldest


_TREE_CACHE = _RSTTreeCache(max_size=512)


# ============================================================================
# RST Parsing
# ============================================================================

def parse_passage_to_rst_tree(passage: str, use_cache: bool = True) -> Optional[Dict]:
    """
    Parse a passage into an RST tree.

    Args:
        passage: Full passage text
        use_cache: Whether to use cached tree if available

    Returns:
        RST tree structure with nodes containing:
        - id: unique node identifier
        - relation: discourse relation type (string)
        - nuclearity: 'NN', 'NS', 'SN' (Nucleus-Nucleus, Nucleus-Satellite, etc.)
        - left, right: child nodes (recursive)
        - start, end: character offsets in passage
        - text: span text

    Returns None on error.
    """
    if not passage or len(passage.strip()) < 5:
        return None

    # Check cache
    if use_cache:
        cached = _TREE_CACHE.get(passage)
        if cached is not None:
            return cached

    try:
        parser = _get_parser()
        result = parser(passage)

        if not result or 'rst' not in result or not result['rst']:
            return None

        tree_root = result['rst'][0]

        # Cache the tree
        _TREE_CACHE.put(passage, tree_root)

        return tree_root

    except Exception as e:
        print(f"[RST Parser] Error parsing passage: {e}")
        return None


def parse_sentence_to_intra_tree(sentence: str) -> Optional[Dict]:
    """
    Parse a sentence as a mini-discourse to extract intra-sentence structure.
    (Intra-sentence trees are lightweight and not cached due to low reuse probability.)

    Args:
        sentence: Single sentence text

    Returns:
        RST tree for sentence (if parseable), else None
    """
    if not sentence or len(sentence.strip()) < 3:
        return None

    try:
        parser = _get_parser()
        result = parser(sentence)

        if not result or 'rst' not in result or not result['rst']:
            return None

        return result['rst'][0]

    except Exception as e:
        print(f"[RST Parser] Error parsing sentence: {e}")
        return None


# ============================================================================
# Tree Navigation Utilities
# ============================================================================


def _node_get(node, key, default=None):
    """Read a field from either a dict node or a DiscourseUnit object."""
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _node_child(node, key):
    return _node_get(node, key, None)

def get_node_depth(node: Dict, current_depth: int = 0) -> int:
    """Recursively compute depth of a node in the RST tree."""
    if node is None:
        return current_depth

    left_node = _node_child(node, 'left')
    right_node = _node_child(node, 'right')
    left_depth = get_node_depth(left_node, current_depth + 1) if left_node is not None else current_depth
    right_depth = get_node_depth(right_node, current_depth + 1) if right_node is not None else current_depth

    return max(left_depth, right_depth)


def get_subtree_size(node: Dict) -> int:
    """Count total nodes in subtree."""
    if node is None:
        return 0
    return 1 + get_subtree_size(_node_child(node, 'left')) + get_subtree_size(_node_child(node, 'right'))


def find_nodes_by_char_range(
    tree: Dict, 
    start: int, 
    end: int
) -> List[Dict]:
    """
    Find all nodes in tree that overlap with character range [start, end).
    Returns list of overlapping nodes.
    """
    if tree is None:
        return []

    overlapping = []

    def traverse(node):
        if node is None:
            return

        node_start = _node_get(node, 'start', -1)
        node_end = _node_get(node, 'end', -1)

        # Check if node overlaps with target range
        if not (node_end <= start or node_start >= end):
            overlapping.append(node)

        left_node = _node_child(node, 'left')
        right_node = _node_child(node, 'right')
        if left_node is not None:
            traverse(left_node)
        if right_node is not None:
            traverse(right_node)

    traverse(tree)
    return overlapping


def get_node_centrality(node: Dict, total_nodes: int) -> float:
    """
    Compute centrality score: (nodes in subtree) / (total nodes in passage).
    Range: [0, 1], where 1 = entire passage (root).
    """
    if total_nodes <= 0:
        return 0.0
    subtree_size = get_subtree_size(node)
    return min(1.0, subtree_size / total_nodes)


def get_node_distance_to_root(node: Dict, tree: Dict) -> int:
    """Compute distance from node to root (0 for root itself, 1 for root's children, etc.)."""

    def find_depth(current: Dict, target_id: int, curr_depth: int) -> Optional[int]:
        if current is None:
            return None

        if _node_get(current, 'id') == target_id:
            return curr_depth

        left_depth = find_depth(_node_child(current, 'left'), target_id, curr_depth + 1)
        if left_depth is not None:
            return left_depth

        right_depth = find_depth(_node_child(current, 'right'), target_id, curr_depth + 1)
        if right_depth is not None:
            return right_depth

        return None

    target_id = _node_get(node, 'id')
    root_id = _node_get(tree, 'id')

    if target_id == root_id:
        return 0

    depth = find_depth(tree, target_id, 0)
    return depth if depth is not None else -1  # -1 if not found


# ============================================================================
# Relation Type Encoding
# ============================================================================

# Common RST relation types (extended list)
RELATION_TYPES = [
    'elaboration',
    'contrast',
    'cause',
    'condition',
    'background',
    'explanation',
    'evidence',
    'concession',
    'result',
    'attribution',
    'evaluation',
    'summary',
    'preparation',
    'restatement',
    'other'
]

RELATION_TO_ID = {rel: idx for idx, rel in enumerate(RELATION_TYPES)}


def encode_relation_type(relation_str: str) -> int:
    """
    Encode discourse relation type as integer (0-14).
    Unknown types map to 'other' (14).
    """
    if not relation_str:
        return 14  # unknown

    relation_lower = relation_str.lower().strip()

    # Direct match
    if relation_lower in RELATION_TO_ID:
        return RELATION_TO_ID[relation_lower]

    # Substring match (e.g., "elaboration-object-property" -> "elaboration")
    for rel in RELATION_TYPES[:-1]:  # Exclude 'other'
        if rel in relation_lower:
            return RELATION_TO_ID[rel]

    return 14  # other/unknown


def get_relation_importance(relation_id: int) -> float:
    """
    Return a crude importance weight for relation type.
    Higher value = more important for salience.
    This is a heuristic; can be refined with task-specific analysis.
    """
    importance_map = {
        0: 0.9,   # elaboration (adds detail, often important)
        1: 0.8,   # contrast (highlights differences)
        2: 0.95,  # cause (causal relations often salient)
        3: 0.7,   # condition (background context)
        4: 0.6,   # background (contextual, less important)
        5: 0.85,  # explanation (clarifies concepts)
        6: 0.9,   # evidence (supports claims)
        7: 0.7,   # concession (acknowledges counter-point)
        8: 0.9,   # result (important outcome)
        9: 0.75,  # attribution (credit/source)
        10: 0.8,  # evaluation (judgment/salience signal)
        11: 0.7,  # summary (recap, less novel)
        12: 0.5,  # preparation (setup, often filler)
        13: 0.6,  # restatement (repetition)
        14: 0.5,  # other (unknown)
    }
    return importance_map.get(relation_id, 0.5)


# ============================================================================
# Sentence-to-EDU Mapping
# ============================================================================

def map_sentence_to_nodes(
    passage: str,
    sentence_text: str,
    sentence_start_char: int,
    tree: Dict
) -> List[Dict]:
    """
    Find RST tree nodes that correspond to a given sentence.

    Args:
        passage: Full passage text
        sentence_text: Text of the sentence
        sentence_start_char: Starting character position of sentence in passage
        tree: RST tree root node

    Returns:
        List of RST nodes that overlap with the sentence's character range.
    """
    if tree is None:
        return []

    sentence_end_char = sentence_start_char + len(sentence_text)
    nodes = find_nodes_by_char_range(tree, sentence_start_char, sentence_end_char)

    return nodes


# ============================================================================
# Intra-Sentence Clause Detection
# ============================================================================

def detect_clause_boundaries(sentence: str) -> List[Tuple[int, int]]:
    """
    Detect clause boundaries in a sentence using simple heuristics.

    Looks for:
    - Coordinate conjunctions: "and", "but", "or", "yet", etc.
    - Subordinate markers: "that", "which", "because", "although", etc.

    Returns:
        List of (start_char, end_char) tuples representing presumed clause spans.
    """
    import re

    COORD_CONJ = r'\b(and|but|or|yet|nor|so)\b'
    SUBORD_MARKERS = r'\b(that|which|who|whom|whose|because|although|though|if|unless|while|when|after|before|since)\b'

    # Find all conjunction positions
    coord_matches = list(re.finditer(COORD_CONJ, sentence, re.IGNORECASE))
    subord_matches = list(re.finditer(SUBORD_MARKERS, sentence, re.IGNORECASE))

    all_breaks = sorted(
        [(m.start(), 'coord') for m in coord_matches] +
        [(m.start(), 'subord') for m in subord_matches]
    )

    if not all_breaks:
        return [(0, len(sentence))]  # Single clause

    # Construct clause spans
    clauses = []
    prev_end = 0

    for pos, break_type in all_breaks:
        if pos > prev_end:
            clauses.append((prev_end, pos))
        prev_end = pos

    # Add final clause
    clauses.append((prev_end, len(sentence)))

    return [(s, e) for s, e in clauses if e > s]  # Filter empty


def count_clauses(sentence: str) -> int:
    """Simple heuristic: count presumed clauses in sentence."""
    clauses = detect_clause_boundaries(sentence)
    return len(clauses)
