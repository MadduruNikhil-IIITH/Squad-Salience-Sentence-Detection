import json

with open('results/qg/run_750_passages_current/qa_generation_issues.json') as f:
    data = json.load(f)

issues = data['issues']

# Get examples 0, 3, 7, 10 to check their differences
for example_idx in [0, 3, 7, 10]:
    example_issues = [issue for issue in issues if issue['example_index'] == example_idx]
    baseline = [issue for issue in example_issues if issue['method'] == 'baseline']
    salience = [issue for issue in example_issues if issue['method'] == 'salience']
    
    if baseline and salience:
        b = baseline[0]
        s = salience[0]
        print(f"\n=== Example {example_idx}: {b['title']} ===")
        print(f"Gold Q: {b['gold_question']}")
        print(f"\nBaseline:")
        print(f"  Q: {b['candidate_question']}")
        print(f"  A: {b['generated_answer']}")
        print(f"  QA F1: {b['qa_consistency_f1']}")
        print(f"\nSalience:")
        print(f"  Q: {s['candidate_question']}")
        print(f"  A: {s['generated_answer']}")
        print(f"  QA F1: {s['qa_consistency_f1']}")
        print(f"\nSalient sentences ({len(s['salient_sentences'])} total):")
        for i, sent in enumerate(s['salient_sentences'][:2]):
            print(f"  {i+1}. {sent[:80]}...")
