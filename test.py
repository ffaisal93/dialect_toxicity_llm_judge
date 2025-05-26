import asyncio
import json
import aiohttp
import os
import argparse
from openai import AsyncOpenAI

# Prompt creator
def make_prompt(eng, trans, tgt_lang):
    return [
        {
            "role": "system",
            "content": (
                "You will be provided with a sentence in English (`src_lang`) along with its translation in another dialect or language (`tgt_lang`).\n\n"
                "Your task is to carefully evaluate the translation:\n\n"
                "- If the translation is accurate, natural, and free of errors, return it exactly as is.\n"
                "- If there are any issues (e.g., incorrect words, unnatural phrasing, repetition, or other mistakes), correct them and provide the improved version.\n\n"
                "Your response should contain only the corrected translation text — no explanations, comments, or extra formatting."
            )
        },
        {
            "role": "user",
            "content": f"src_lang: English\ntgt_lang: {tgt_lang}\nEnglish: {eng}\nTranslation: {trans}"
        }
    ]

# Async function to send a single request
async def correct_translation(client, model, eng, trans, tgt_lang, semaphore, idx):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=make_prompt(eng, trans, tgt_lang),
                temperature=0.7,
            )
            return {"index": idx, "corrected": response.choices[0].message.content.strip()}
        except Exception as e:
            print(f"Error on entry {idx}: {e}")
            return {"index": idx, "corrected": trans}

# Main async pipeline
async def process_all_translations(client, model, data, max_concurrency=5):
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        correct_translation(client, model, entry["english"], entry["translation"], entry["tgt_lang"], semaphore, idx)
        for idx, entry in enumerate(data)
    ]
    results = await asyncio.gather(*tasks)
    return results

# Entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key_file', required=True)
    parser.add_argument('--input_json', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--model_name', default="gpt-4o")
    parser.add_argument('--max_concurrency', type=int, default=5)
    args = parser.parse_args()

    with open(args.api_key_file, "r") as f:
        api_key = f.read().strip()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = AsyncOpenAI(api_key=api_key)

    # Run async event loop
    results = asyncio.run(process_all_translations(client, args.model_name, data, args.max_concurrency))

    # Merge back with original data
    for result in results:
        idx = result["index"]
        data[idx]["corrected"] = result["corrected"]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ All done. Output saved to {args.output_json}")

if __name__ == "__main__":
    main()
