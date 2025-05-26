import os
import json
import argparse
import logging
import asyncio
from openai import AsyncOpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Dialect keys you're interested in
target_dialects = {
    "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab", "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab",
     "standard_Bn",
    "yue_Hant", "zho_Hans", "zho_Hant",
    "tur_Latn", "azb_Arab", "azj_Latn",
    "ltg_Latn", "lvs_Latn",
    "ckb_Arab", "kmr_Latn",
    "nno_Latn", "nob_Latn",
    "nso_Latn", "sot_Latn"
}
# target_dialects = {
#     # "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab", "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab",
#     "standard_Bn",
#     # "yue_Hant", "zho_Hans", "zho_Hant",
#     # "tur_Latn", "azb_Arab", "azj_Latn",
#     # "ltg_Latn", "lvs_Latn",
#     # "ckb_Arab", "kmr_Latn",
#     # "nno_Latn", "nob_Latn",
#     # "nso_Latn", "sot_Latn"
# }

# Prompt template
def create_prompt(src_lang, tgt_lang, english, translation):
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
            "content": f"src_lang: {src_lang}\ntgt_lang: {tgt_lang}\nEnglish: {english}\nTranslation: {translation}"
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]

async def correct_translation(client, model_name, eng, trans, tgt_lang, semaphore, i, key, fname):
    prompt = create_prompt("English", tgt_lang, eng, trans)
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=prompt,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in {fname} — {key} — index {i}: {e}")
            return trans

async def process_file(client, args, lang_mapping, gold_texts, fname):
    file_path = os.path.join(args.data_path, fname)
    with open(file_path, "r", encoding="utf-8") as f:
        lang_data = json.load(f)

    modified = False
    semaphore = asyncio.Semaphore(args.max_concurrency)

    for key in lang_data:
        if key not in target_dialects:
            continue
        tgt_lang = lang_mapping.get(key, key)
        translations = lang_data[key]

        if len(translations) != len(gold_texts):
            logger.warning(f"⚠️ Skipping {fname} — {key} due to length mismatch "
                           f"({len(translations)} translations vs {len(gold_texts)} gold)")
            continue

        logger.info(f"Processing: {fname} — Dialect: {key} ({tgt_lang})")

        tasks = [
            correct_translation(client, args.model_name, eng, trans, tgt_lang, semaphore, i, key, fname)
            for i, (eng, trans) in enumerate(zip(gold_texts, translations))
        ]
        corrected_translations = await asyncio.gather(*tasks)
        lang_data[key] = corrected_translations
        modified = True

    if modified:
        out_path = os.path.join(args.output_path, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lang_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved corrected file: {out_path}")

async def main():
    parser = argparse.ArgumentParser(description="Translation correction pipeline using GPT (Async)")
    parser.add_argument('--api_key_file', required=True, help="Path to the file containing the OpenAI API key")
    parser.add_argument('--data_path', default="data/processed_data", help="Path to the input data folder")
    parser.add_argument('--output_path', default="data/processed_data_gpt_assisted", help="Path to save corrected files")
    parser.add_argument('--lang_map_path', default="metadata/lang_mapping.json", help="Path to the language mapping JSON")
    parser.add_argument('--model_name', default="gpt-4o", help="GPT model name to use")
    parser.add_argument('--max_concurrency', type=int, default=5, help="Number of concurrent requests")
    args = parser.parse_args()

    with open(args.api_key_file, "r") as f:
        api_key = f.read().strip()

    client = AsyncOpenAI(api_key=api_key)

    with open(args.lang_map_path, "r", encoding="utf-8") as f:
        lang_mapping = json.load(f)

    with open(os.path.join(args.data_path, "english.json"), "r", encoding="utf-8") as f:
        english_data = json.load(f)
    gold_texts = english_data.get("standard", [])

    os.makedirs(args.output_path, exist_ok=True)

    tasks = []
    for fname in os.listdir(args.data_path):
        if not fname.endswith(".json") or fname == "english.json":
            continue
        tasks.append(process_file(client, args, lang_mapping, gold_texts, fname))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())