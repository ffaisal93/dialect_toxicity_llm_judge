import os
import json
import argparse
import ctranslate2
import transformers
from tqdm import tqdm

src_languages = {
    "acm_Arab", "acq_Arab", "aeb_Arab", "ajp_Arab", "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab",
    "standard_Bn",
    "yue_Hant", "zho_Hans", "zho_Hant",
    "tur_Latn", "azb_Arab", "azj_Latn",
    "ltg_Latn", "lvs_Latn",
    "ckb_Arab", "kmr_Latn",
    "nno_Latn", "nob_Latn",
    "nso_Latn", "sot_Latn"
}

# Translation function
def translate_to_english(translator, tokenizer, sentences, src_lang):
    batch_size = 128
    translations = []
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Translating {src_lang}"):
        batch = sentences[i:i + batch_size]
        tokenized = tokenizer(batch)
        source = [tokenizer.convert_ids_to_tokens(sent) for sent in tokenized["input_ids"]]
        target_prefix = [["eng_Latn"]] * len(batch)

        try:
            results = translator.translate_batch(source, target_prefix=target_prefix, beam_size=1)
            token_ids = [tokenizer.convert_tokens_to_ids(r.hypotheses[0][1:]) for r in results]
            decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            translations.extend(decoded)
        except Exception as e:
            print(f"Error translating {src_lang} batch {i}: {e}")
            translations.extend(["" for _ in batch])

    return translations

def main():
    parser = argparse.ArgumentParser(description="Translate dialect files to English using NLLB-CT2")
    parser.add_argument("--data_path", default="data/processed_data", help="Directory with input JSON files")
    parser.add_argument("--output_path", default="data/translated_to_english", help="Directory to save translated output")
    parser.add_argument("--model_path", default="models/ct2fast-nllb-200-3.3B", help="Path to the CTranslate2 model")
    parser.add_argument("--device", default="cpu", help="Device to use for translation")
    parser.add_argument("--output_file_name", default="nllb.json", help="Name of the output JSON file")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    translator = ctranslate2.Translator(args.model_path, device=args.device)
    final_translations = {}

    for fname in os.listdir(args.data_path):
        if not fname.endswith(".json"):
            continue

        file_path = os.path.join(args.data_path, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for dialect_key, sentences in data.items():
            if dialect_key not in src_languages:
                continue

            model_lang = "ben_Beng" if dialect_key == "standard_Bn" else dialect_key
            tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=model_lang)
            translations = translate_to_english(translator, tokenizer, sentences, model_lang)
            final_translations[dialect_key] = translations

    # Save combined output
    output_file = os.path.join(args.output_path, args.output_file_name)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_translations, f, ensure_ascii=False, indent=2)

    print(f"✅ All translations saved to {output_file}")

if __name__ == "__main__":
    main()