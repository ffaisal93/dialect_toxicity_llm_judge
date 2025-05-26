import os
import json
import argparse
import evaluate
from collections import defaultdict

def compute_bleu_scores(predictions, references):
    bleu = evaluate.load("sacrebleu")
    return bleu.compute(predictions=predictions, references=references)

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLEU scores for translated output against gold English references")
    parser.add_argument("--translated_file", default="data/translated_to_english/nllb.json", help="Path to the translated JSON file")
    parser.add_argument("--reference_file", default="data/processed_data/english.json", help="Path to the gold standard English JSON file")
    parser.add_argument("--result_file", default="results/bleu_scores.json", help="Path to save BLEU score dictionary (JSON)")

    args = parser.parse_args()

    # Load data
    with open(args.translated_file, "r", encoding="utf-8") as f:
        translated_data = json.load(f)

    with open(args.reference_file, "r", encoding="utf-8") as f:
        reference_data = json.load(f)
    references_all = reference_data.get("standard", [])

    dialect_scores = {}
    all_preds = []
    all_refs = []

    for dialect, predictions in translated_data.items():
        if len(predictions) != len(references_all):
            print(f"Skipping {dialect}: length mismatch ({len(predictions)} vs {len(references_all)})")
            continue

        references = [[ref] for ref in references_all]
        result = compute_bleu_scores(predictions, references)
        dialect_scores[dialect] = result["score"]
        all_preds.extend(predictions)
        all_refs.extend(references)

    global_result = compute_bleu_scores(all_preds, all_refs)

    print("\n--- BLEU Scores ---")
    for dialect, score in sorted(dialect_scores.items()):
        print(f"{dialect}: {score:.2f}")
    print(f"\nGlobal BLEU score: {global_result['score']:.2f}")

    # Save results to JSON
    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump({
            "dialect_scores": dialect_scores,
            "global_bleu": global_result["score"]
        }, f, indent=2)

if __name__ == "__main__":
    main()
