import sacrebleu
from nllbseed import NllbSeedData
from tqdm import tqdm
import os
import csv
from datetime import datetime
import argparse
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer
from configure import *
from multilingualdata import * 


def translate(
    text, tokenizer, model,
    src_lang, tgt_lang,
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    model.eval() # turn off training mode
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True,
        max_length=max_input_length
    )
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def batched_translate(texts, batch_size=16, **kwargs):
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]


def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result = bleu_calc.compute(predictions = candidate_translations, references = reference_translations)
    bleu_string = str(bleu_result)
    print(bleu_string)
    chrf_result = chrf_calc.compute(predictions = candidate_translations, references = reference_translations)
    chrf_string = str(chrf_result)
    print(chrf_string)
    return round(bleu_result["score"], 3), round(chrf_result["score"], 3)

def log_evaluation(log_file, model_name, target_lang, bleu, chrf, notes):

    file_exists = os.path.isfile(log_file)

    # Open the log file in append mode
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header if the file is empty
        if not file_exists:
            writer.writerow(['model_name', 'date', 'target_lang', 'BLEU', 'ChrF++', 'notes'])

        # Write the new row with the evaluation data
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow(
            [model_name,
             date,
             target_lang,
             bleu,
             chrf,
             notes])


###TODO: fix this evaluation so that it logs as well
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluation script for NLLB models.")
    parser.add_argument("--data", type=str, required=True, choices=['nllb-seed', 'americas-nlp'], help="Finetuning data")
    parser.add_argument("--model_dir", type=str, help="Directory for storing the trained model")
    parser.add_argument("--tgt", type=str, required=True, help="Target language for validation.")

    args = parser.parse_args()
    model_dir = args.model_dir
    tgt = args.tgt
    
    # load in pretrained models
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).cuda()
    tokenizer = NllbTokenizer.from_pretrained(model_dir)

    csv_file = NLLB_SEED_CSV if args.data == 'nllb-seed' else AMERICAS_NLP_CSV
    
    # create lanauge pairs for evaluatuon
    if tgt == "multi":
        lps = [('spa_Latn', lang) for lang in ALL_AMERICAS_NLP_LANGS]
    else:
        lps = [('spa_Latn', tgt)]
    
    corpus = MultilingualCorpus(csv_file)

    print('Starting logging to ', LOG_FILE)
    notes = TRAINING_NOTES # get notes from configure.py
    for src, tgt in lps:
        try:
            print('Evaluating ', model_dir.split('/')[-1], 'on: ', src, '-->', tgt)
            eval_bitext = corpus.create_bitext(src, tgt, 'dev')
            src_texts, tgt_texts = eval_bitext.lang1_sents, eval_bitext.lang2_sents
            candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=eval_bitext.lang1_code, tgt_lang=eval_bitext.lang2_code)
            bleu, chrf = evaluate_translations(candidate_translations, tgt_texts)
            tgt_lang = AMERICAS_NLP_CODE_TO_LANG[tgt.split('_')[0]]
            log_evaluation(LOG_FILE, model_dir.split('/')[-1], tgt_lang, bleu, chrf, notes) # log a line for each evaluation
            print('Wrote all evaluations to ', LOG_FILE)
        except Exception as e:
            print(f"Error occurred while evaluating {src} --> {tgt}: {str(e)}")