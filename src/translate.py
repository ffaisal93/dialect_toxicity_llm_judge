# Load the model and tokenizer
# Important: This should be done only once

import ctranslate2
import transformers
import os
import pandas as pd
import pickle

# Arabic dialects and other target languages
target_languages = [
    "arb_Arab", "acq_Arab", "ars_Arab", "acm_Arab", "ajp_Arab", "apc_Arab", "arz_Arab", "aeb_Arab", "ary_Arab",
    "swe_Latn", "fin_Latn", "ben_Beng"
]

target_languages = ['azj_Latn','tur_Latn','azb_Arab','ckb_Arab','kmr_Latn','lvs_Latn','ltg_Latn','nob_Latn','nno_Latn','zho_Hans','zho_Hant','yue_Hant','nso_Latn','sot_Latn']

src_lang = "eng_Latn"
device = "cuda"  # or "cpu"
beam_size = 4

# Initialize the translator and tokenizer
translator = ctranslate2.Translator("models/ct2fast-nllb-200-3.3B", device)
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=src_lang)

# Load source sentences from dataset
df = pd.read_parquet('data/toxigen/test-00000-of-00001.parquet')
source_sents = df['text'].tolist()

# Iterate over each target language
for tgt_lang in target_languages:
    dest_file = f"data/nllb_toxigen_test/{tgt_lang}.pkl"
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    all_translations = []

    for i in range(0, len(source_sents), 64):
        print(f"Translating to {tgt_lang}: {i}, {round((i * 100) / len(source_sents))}% complete")
        source_sents_batch = source_sents[i:min(i + 64, len(source_sents))]
        source_sents_tokenized = tokenizer(source_sents_batch)
        source = [tokenizer.convert_ids_to_tokens(sent) for sent in source_sents_tokenized["input_ids"]]
        target_prefix = [[tgt_lang]] * len(source_sents_batch)
        
        try:
            results = translator.translate_batch(source, target_prefix=target_prefix, beam_size=beam_size)
            target_sents_tokenized = [result.hypotheses[0][1:] for result in results]
            target_sents_to_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in target_sents_tokenized]
            translations = tokenizer.batch_decode(target_sents_to_ids)
            all_translations.extend(translations)
        except Exception as e:
            print(f"Error translating batch {i} for {tgt_lang}: {e}")
            all_translations.extend(["" for _ in source_sents_batch])  # Add empty strings for failed translations

    # Save translations to pickle file
    with open(dest_file, 'wb') as f:
        pickle.dump(all_translations, f)

    print(f"Translations for {tgt_lang} saved to {dest_file}")