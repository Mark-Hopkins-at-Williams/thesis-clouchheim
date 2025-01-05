from finetune import *
from multilingualdata import MixtureOfBitexts

# main method to add a monolingual (bitext <gen> --> quy_Latn) to the data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning script for NLLB models.")
    parser.add_argument("--data", type=str, required=True, choices=['nllb-seed', 'americas-nlp'], help="Finetuning data")
    parser.add_argument("--model_dir", type=str, help="Directory for storing the trained model")
    parser.add_argument("--nllb_model", type=str, default="600M", choices=['600M', '1.3B', '3.3B', 'other'], help="NLLB base model.")
    parser.add_argument("--dev_src", type=str, required=True, help="Source language for validation.")
    parser.add_argument("--dev_tgt", type=str, required=True, help="Target language for validation.")
    parser.add_argument("--other_model", type=str, required=False, help="Finetune model")
    
    args = parser.parse_args()
    model_dir = args.model_dir
    if args.nllb_model != 'other':
        model_name = "facebook/nllb-200-distilled-" + args.nllb_model
    else:
        model_name = args.other_model
    
    if os.path.exists(model_dir):
        print(f"model directory already exists: {model_dir}")
        exit()
        
    csv_file = NLLB_SEED_CSV if args.data == 'nllb-seed' else AMERICAS_NLP_CSV
    lps = NLLB_SEED_LPS if args.data == 'nllb-seed' else AMERICAS_NLP_LPS ### this is where I can add the <generate>
    corpus = MultilingualCorpus(csv_file)
    gen_corpus = MultilingualCorpus('data/monolingual_quechua_train_cleaned.csv') # TODO: generalize to other languages
    
    bitexts = []
    for lp in lps:
        bitexts.append(corpus.create_bitext(lp[0], lp[1], 'train'))
    bitexts.append(gen_corpus.create_gen('quy_Latn')) # TODO: generalize to other languages
    
    train_data = MixtureOfBitexts(bitexts, batch_size = 8, sampling_probs = [0.9, 0.1]) # sample parallel 90% and <gen> 10% 
    dev_bitext = [corpus.create_bitext(args.dev_src, args.dev_tgt, split = 'dev')]
    future_eval_lps = AMERICAS_NLP_LPS
    notes = TRAINING_NOTES # get notes from configure.py

    print('Training ', model_dir)
    print('Langs in training:', train_data.get_language_codes())
    model, tokenizer = finetune(train_data, dev_bitext, model_name, model_dir)
    print('Done training ', model_dir)

    # TODO: automatic logging system not working
    print('Starting logging to ', LOG_FILE)
    for src, tgt in future_eval_lps:
        try:
            print('Evaluating: ', src, '-->', tgt)
            eval_bitext = corpus.create_bitext(src, tgt, 'dev')
            src_texts, tgt_texts = eval_bitext.lang1_sents, eval_bitext.lang2_sents   
            candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=eval_bitext.lang1_code, tgt_lang=eval_bitext.lang2_code)
            bleu, chrf = evaluate_translations(candidate_translations, tgt_texts)
            tgt_lang = AMERICAS_NLP_CODE_TO_LANG[tgt.split('_')[0]]
            log_evaluation(LOG_FILE, model_dir.replace("models/", ""), tgt_lang, bleu, chrf, notes) # log a line for each evaluation
            print('Wrote all evaluations to ', LOG_FILE)
        except Exception as e:
            print(f"Error occurred while evaluating {src} --> {tgt}: {str(e)}")
        
    