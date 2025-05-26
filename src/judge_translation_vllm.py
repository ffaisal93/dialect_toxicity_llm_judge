import os
import json
import argparse
import random
from vllm import LLM, SamplingParams
from collections import defaultdict
from tqdm import tqdm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to judge model")
    parser.add_argument("--gold_file", default="data/processed_data/english.json")
    parser.add_argument("--translation_a_file", default="data/translated_to_english/nllb.json")
    parser.add_argument("--translation_b_file", default="data/translated_to_english/nllb_gpt_assisted.json")
    args = parser.parse_args()

    # Load LLM
    llm = LLM(model=args.model_path)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2)

    # Load data
    gold_data = load_json(args.gold_file)["standard"]
    data_a = load_json(args.translation_a_file)
    data_b = load_json(args.translation_b_file)

    dialect_scores = defaultdict(lambda: {"a": 0, "b": 0, "both": 0, "neither": 0, "count": 0})
    cc = 0

    for dialect in tqdm(data_a.keys(), desc="Evaluating"):
        cc += 1
        if cc > 2:
            break
        if dialect not in data_b:
            continue
        trans_a = data_a[dialect]
        trans_b = data_b[dialect]

        if len(trans_a) != len(trans_b) or len(trans_a) != len(gold_data):
            print(f"Skipping {dialect}: length mismatch")
            continue

        prompts = []
        mappings = []
        for gold, a_bt, b_bt in zip(gold_data, trans_a, trans_b):
            if random.random() < 0.5:
                candidate1, candidate2 = a_bt, b_bt
                mapping = {"X": "a", "Y": "b"}
            else:
                candidate1, candidate2 = b_bt, a_bt
                mapping = {"X": "b", "Y": "a"}
            mappings.append(mapping)

            prompt = (
                "You must respond with exactly one of these four options: X, Y, XY, or 0.\n"
                "Do not include any other text. No explanation.\n\n"
                "Which translation is more similar in meaning, coherence, and naturalness to the reference sentence?\n"
                "- XY: Both are acceptable\n"
                "- X: Only X is acceptable\n"
                "- Y: Only Y is acceptable\n"
                "- 0: Neither is acceptable\n\n"
                f"Reference: {gold}\n"
                f"Candidate X: {candidate1}\n"
                f"Candidate Y: {candidate2}\n"
                "Answer:"
            )
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)

        for response, mapping in zip(outputs, mappings):
            answer = response.outputs[0].text.strip().upper()

            if answer == "XY":
                dialect_scores[dialect]["both"] += 1
            elif answer in mapping:
                dialect_scores[dialect][mapping[answer]] += 1
            else:
                dialect_scores[dialect]["neither"] += 1

            dialect_scores[dialect]["count"] += 1

    print("\n--- Judge Evaluation Scores (normalized by total count) ---")
    total_a = total_b = total_both = total_neither = total_count = 0
    for dialect, score in sorted(dialect_scores.items()):
        count = score["count"]
        if count == 0:
            continue
        a_ratio = score["a"] / count
        b_ratio = score["b"] / count
        both_ratio = score["both"] / count
        neither_ratio = score["neither"] / count
        win = "A" if a_ratio > b_ratio else "B" if b_ratio > a_ratio else "Tie"
        print(f"{dialect}: A = {a_ratio:.2f}, B = {b_ratio:.2f}, BOTH = {both_ratio:.2f}, NEITHER = {neither_ratio:.2f}, WIN = {win}")
        total_a += score["a"]
        total_b += score["b"]
        total_both += score["both"]
        total_neither += score["neither"]
        total_count += count

    print("\n--- GLOBAL Scores ---")
    print(f"GLOBAL: A = {total_a / total_count:.2f}, B = {total_b / total_count:.2f}, BOTH = {total_both / total_count:.2f}, NEITHER = {total_neither / total_count:.2f}")

if __name__ == "__main__":
    main()