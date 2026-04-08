import nltk
import re
import numpy as np
from collections import Counter
from surprisal import get_surprisal_features

try:
    import rst_parser
    RST_AVAILABLE = True
except ImportError:
    RST_AVAILABLE = False


def _node_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)

RE_CLEAN = re.compile(r"[^a-zA-Z\s]")
RE_SPACE = re.compile(r"\s+")
CAUSAL_MARKERS = {"because", "since", "as", "so", "therefore", "thus", "hence"}
CONTRAST_MARKERS = {"but", "however", "although", "though", "yet", "still", "nevertheless"}

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = RE_CLEAN.sub(" ", text)
    return RE_SPACE.sub(" ", text).strip().lower()

def extract_linguistic_features(
    sentence: str,
    position: int,
    total_sentences: int,
    include_surprisal: bool = True,
    include_rst: bool = False,
    full_passage: str = "",
    sent_index: int = -1
) -> dict:
    original = sentence.strip()
    cleaned = clean_text(original) or "empty"
    
    tokens = nltk.word_tokenize(cleaned)
    tagged = nltk.pos_tag(tokens, tagset="universal")
    
    words = [t for t in tokens if t.isalpha()]
    if not words:
        words = ["x"]
    
    word_lengths = [len(w) for w in words]
    types = set(words)
    pos_counts = Counter(p for _, p in tagged)
    total_pos = sum(pos_counts.values()) or 1
    total_tokens = len(tokens) or 1

    ne_count = sum(1 for w in original.split() if w.isalpha() and w[0].isupper() and len(w)>1)

    ling_features = {
        "avg_word_length": np.mean(word_lengths),
        "sentence_length_words": len(words),
        "sentence_position": position,
        "sentence_position_norm": position / total_sentences,
        "type_token_ratio": len(types) / len(words),
        "lexical_density": sum(1 for _,p in tagged if p in {"NOUN","VERB","ADJ","ADV"}) / len(words),
        "noun_ratio": pos_counts["NOUN"]/total_pos,
        "verb_ratio": pos_counts["VERB"]/total_pos,
        "adj_ratio": pos_counts["ADJ"]/total_pos,
        "pronoun_ratio": pos_counts["PRON"]/total_pos,
        "noun_verb_ratio": pos_counts["NOUN"]/(pos_counts["VERB"]+1e-8),
        "causal_marker_ratio": sum(w in CAUSAL_MARKERS for w in tokens)/total_tokens,
        "contrast_marker_ratio": sum(w in CONTRAST_MARKERS for w in tokens)/total_tokens,
        "named_entity_density": ne_count / len(words),
    }

    # === OPTIONAL FUTURE FEATURES (kept commented for future experiments) ===
    # 1. Readability (simple proxy)
    # avg_sent_len = len(words)
    # avg_word_len = np.mean(word_lengths)
    # flesch = 206.835 - 1.015 * avg_word_len - 84.6 * (1 / avg_sent_len) if avg_sent_len > 0 else 0
    # flesch = np.clip(flesch, 0, 100)

    # 2. Punctuation density
    # punct_count = len(re.findall(r"[.,!?;:]", original))
    # punctuation_density = punct_count / len(words) if len(words) > 0 else 0

    # 3. Character diversity (char-level TTR proxy)
    # char_types = set(original.lower())
    # char_ttr = len(char_types) / len(original) if original else 0

    # 4. Extended POS ratio
    # adverb_ratio = pos_counts["ADV"] / (pos_counts["NOUN"] + 1e-8)

    # genre_features = {
    #     # "flesch_reading_ease": flesch,
    #     # "punctuation_density": punctuation_density,
    #     # "char_diversity_ttr": char_ttr,
    #     # "adverb_noun_ratio": adverb_ratio,
    # }

    features = dict(ling_features)
    
    if include_surprisal:
        surp = get_surprisal_features(original)
        features.update(surp)
    
    # Add RST features if requested
    if include_rst and RST_AVAILABLE:
        # Passage-level RST features
        if full_passage and sent_index >= 0:
            rst_passage_features = extract_rst_features_passage_level(
                sentence, sent_index, total_sentences, full_passage
            )
            features.update(rst_passage_features)
        else:
            # Missing passage context; add default RST features
            default_rst = {
                "rst_depth": 0.0,
                "rst_is_nucleus": 0.0,
                "rst_relation_type_encoded": 14.0,
                "rst_centrality_score": 0.0,
                "rst_is_root": 0.0,
                "rst_relation_direction": 0.0,
                "rst_span_length": 0.0,
            }
            features.update(default_rst)

        # Intra-sentence RST features
        rst_intra_features = extract_rst_features_intra_sentence(sentence)
        features.update(rst_intra_features)
    
    # Safety
    for k in features:
        if isinstance(features[k], float) and (np.isnan(features[k]) or np.isinf(features[k])):
            features[k] = 0.0

    return features


# ============================================================================
# RST (Rhetorical Structure Theory) Feature Extraction
# ============================================================================

def extract_rst_features_passage_level(
    sentence: str,
    sentence_index: int,
    total_sentences: int,
    full_passage: str
) -> dict:
    """
    Extract RST features at passage level by parsing full passage as discourse.
    Maps the sentence to its corresponding RST tree node(s) and extracts features.

    Returns dict with 7 RST passage-level features:
    - rst_depth: Distance from root node (int, 0-indexed)
    - rst_is_nucleus: 1 if nucleus in parent relation, 0 if satellite (binary)
    - rst_relation_type_encoded: Discourse relation type (0-14, ordinal encoding)
    - rst_centrality_score: Subtree size / total EDUs (float, [0,1])
    - rst_is_root: 1 if sentence is passage root, 0 else (binary)
    - rst_relation_direction: +1 nucleus, 0 unknown, -1 satellite (int)
    - rst_span_length: EDUs in subtree / total (float, [0,1])
    """
    default_features = {
        "rst_depth": 0.0,
        "rst_is_nucleus": 0.0,
        "rst_relation_type_encoded": 14.0,
        "rst_centrality_score": 0.0,
        "rst_is_root": 0.0,
        "rst_relation_direction": 0.0,
        "rst_span_length": 0.0,
    }

    if not RST_AVAILABLE or not full_passage or not sentence:
        return default_features

    try:
        # Parse passage to RST tree
        tree = rst_parser.parse_passage_to_rst_tree(full_passage)
        if tree is None:
            return default_features

        # Calculate character offsets for sentence (approximate via NLTK tokenization)
        sentences = nltk.sent_tokenize(full_passage)
        char_pos = 0
        sentence_start = None
        sentence_end = None

        for sent in sentences:
            if sent.strip() == sentence.strip():
                sentence_start = char_pos
                sentence_end = char_pos + len(sent)
                break
            char_pos += len(sent) + 1  # +1 for space

        if sentence_start is None:
            return default_features

        # Find nodes overlapping with sentence
        nodes = rst_parser.map_sentence_to_nodes(full_passage, sentence, sentence_start, tree)
        if not nodes:
            return default_features

        # Aggregate features from all overlapping nodes
        # (Primary: use shallowest/most specific node)
        best_node = min(nodes, key=lambda n: rst_parser.get_node_depth(n)) if nodes else None

        if best_node is None:
            return default_features

        # Extract features from best_node
        total_tree_nodes = rst_parser.get_subtree_size(tree)
        depth = rst_parser.get_node_distance_to_root(best_node, tree)
        if depth < 0:
            depth = 0

        # Nuclearity: parse 'nuclearity' field (e.g., 'NS', 'SN', 'NN')
        nuclearity_str = _node_get(best_node, 'nuclearity', '')  # first char = left, second = right
        is_nucleus = 1.0 if nuclearity_str and nuclearity_str[0] == 'N' else 0.0

        # Relation type
        relation_str = _node_get(best_node, 'relation', '')
        relation_encoded = float(rst_parser.encode_relation_type(relation_str))

        # Centrality
        centrality = rst_parser.get_node_centrality(best_node, total_tree_nodes)

        # Is root?
        is_root = 1.0 if _node_get(best_node, 'id') == _node_get(tree, 'id') else 0.0

        # Relation direction (N=+1, S=-1, unclear=0)
        relation_direction = 1.0 if is_nucleus > 0 else -1.0 if nuclearity_str and len(nuclearity_str) > 0 else 0.0

        # Span length (relative)
        span_length = rst_parser.get_node_centrality(best_node, total_tree_nodes)

        return {
            "rst_depth": float(depth),
            "rst_is_nucleus": is_nucleus,
            "rst_relation_type_encoded": relation_encoded,
            "rst_centrality_score": centrality,
            "rst_is_root": is_root,
            "rst_relation_direction": relation_direction,
            "rst_span_length": span_length,
        }

    except Exception as e:
        print(f"[Feature Extractor] RST passage-level error: {e}")
        return default_features


def extract_rst_features_intra_sentence(sentence: str) -> dict:
    """
    Extract intra-sentence RST features by parsing sentence as mini-discourse.
    Captures clause-level structure and complexity.

    Returns dict with 5 intra-sentence RST features:
    - intra_clause_count: Number of clauses detected (int)
    - intra_clause_density: clauses / words (float, [0,1])
    - intra_has_nucleus_satellite_relation: 1 if N-S relation exists, 0 else (binary)
    - intra_avg_clause_depth: Average depth of clause nodes (float)
    - intra_top_clause_is_nucleus: 1 if main clause is nucleus-like (binary)
    """
    default_features = {
        "intra_clause_count": 0.0,
        "intra_clause_density": 0.0,
        "intra_has_nucleus_satellite_relation": 0.0,
        "intra_avg_clause_depth": 0.0,
        "intra_top_clause_is_nucleus": 0.0,
    }

    if not RST_AVAILABLE or not sentence:
        return default_features

    try:
        # Heuristic clause counting (lexical approach)
        clause_count = rst_parser.count_clauses(sentence)
        words = nltk.word_tokenize(sentence)
        word_count = len([w for w in words if w.isalpha()]) or 1

        clause_density = min(1.0, clause_count / word_count) if word_count > 0 else 0.0

        # Try parsing sentence as mini-tree for deeper structure
        tree = rst_parser.parse_sentence_to_intra_tree(sentence)

        has_ns_relation = 0.0
        avg_depth = 0.0
        is_root_nucleus = 0.0

        if tree is not None:
            # Check for N-S relations
            nuclearity = _node_get(tree, 'nuclearity', '')
            has_ns_relation = 1.0 if nuclearity in ['NS', 'SN'] else 0.0

            # Average depth of tree
            max_depth = rst_parser.get_node_depth(tree)
            avg_depth = max_depth / 3.0 if max_depth > 0 else 0.0  # Normalize

            # Root nuclearity
            is_root_nucleus = 1.0 if nuclearity and nuclearity[0] == 'N' else 0.0

        return {
            "intra_clause_count": float(clause_count),
            "intra_clause_density": clause_density,
            "intra_has_nucleus_satellite_relation": has_ns_relation,
            "intra_avg_clause_depth": avg_depth,
            "intra_top_clause_is_nucleus": is_root_nucleus,
        }

    except Exception as e:
        print(f"[Feature Extractor] RST intra-sentence error: {e}")
        return default_features