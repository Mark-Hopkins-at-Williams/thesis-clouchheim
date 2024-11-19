import gc
import os
import sys
import csv
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from nllbseed import NllbSeedData
from validate import log_evaluation, batched_translate, evaluate_translations
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer
from configure import USE_CUDA
from configure import AMERICAS_NLP_LANGS, AMERICAS_NLP_CSV, AMERICAS_NLP_LPS, AMERICAS_NLP_CODE_TO_LANG, ALL_LANGS
from configure import NLLB_SEED_CSV, NLLB_SEED_LPS
from configure import LOG_FILE, TRAINING_NOTES
from multilingualdata import MultilingualCorpus, StreamingMonolingual, Monolingual


def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def tokenize(sents, lang, tokenizer, max_length, alt_pad_token=None):
    tokenizer.src_lang = lang
    tokens = tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    if alt_pad_token is not None:
        tokens.input_ids[tokens.input_ids == tokenizer.pad_token_id] = alt_pad_token  # e.g., -100 is a magic value ignored
                                                                                      # in the loss function because we don't want the model to learn to predict padding ids
    return tokens

def tokenizer_texts(opus_corpus, opus_files, anlp_corpus, num):
    # Initialize the monolingual stream list
    mono_list = []
    
    # Add streaming instances for 'eng_Latn' and 'spa_Latn'
    mono_list.append(opus_corpus.create_monolingual('eng_Latn', stream_file=opus_files[0]))
    mono_list.append(opus_corpus.create_monolingual('spa_Latn', stream_file=opus_files[1]))
    
    # Append non-streaming instances for the other languages
    for lang in ALL_LANGS:
        mono_list.append(anlp_corpus.create_monolingual(lang))
    
    def tok_stream():
        for _ in range(num):
            # Randomly select an index from the list
            data = random.choice(mono_list)
            
            # Handle StreamingMonolingual and Monolingual objects differently
            if isinstance(data, StreamingMonolingual):
                # For StreamingMonolingual, yield the next sentence from the stream
                try:
                    yield next(iter(data))
                except StopIteration:
                    continue  # If the stream ends, continue to the next iteration
            elif isinstance(data, Monolingual):
                # For Monolingual, yield the next sentence from the list
                try:
                    yield data.get_next()
                except StopIteration:
                    continue  # If there are no more sentences, continue to the next iteration
                
    return tok_stream()


def finetune(mixture_of_bitexts, dev_bitext_list, base_model, finetuned_model_dir,
             training_steps=60000,
             max_length=128, # token sequences will be truncated to this many tokens
             report_every=100,
             validate_every=1000, 
             sampling_probs = None
             ):
    
    # set up sampling probs for dev
    if sampling_probs:
        sampling_probs = [p / sum(sampling_probs) for p in sampling_probs]
    else:
        sampling_probs = [1 / len(dev_bitext_list)] * len(dev_bitext_list)
    
    # load model and tokenizer
    if type(base_model) is list: # this is what you do if it is pretrained
        model = base_model[0]
        tokenizer = base_model[1]
    elif 'facebook' not in base_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model, local_files_only=True).cuda()
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    print('Done loading in model and tokenzier')
        
    new_lang_codes = [code for code in mixture_of_bitexts.get_language_codes() if code in tokenizer.get_vocab()]
    tokenizer.add_tokens(new_lang_codes)
    model.resize_token_embeddings(len(tokenizer))
    
    if USE_CUDA:
        model.cuda()
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
    x, y, train_loss = None, None, None
    last_best = 0
    last_best_chrf = 0
    patience = training_steps * 0.5
    cleanup()
    train_losses = []   # tracks average loss
    for i in tqdm(range(training_steps)):
        sys.stdout.flush()
        lang1_sents, lang2_sents, lang1, lang2 = mixture_of_bitexts.next_batch()
        try:
            model.train()
            x = tokenize(lang1_sents, lang1, tokenizer, max_length).to(model.device)
            y = tokenize(lang2_sents, lang2, tokenizer, max_length, alt_pad_token=-100).to(model.device)
            train_loss = model(**x, labels=y.input_ids).loss
            train_loss.backward()
            train_losses.append(train_loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        except RuntimeError:  # handle GPU-out-of-memory exceptions
            optimizer.zero_grad(set_to_none=True)
            x, y, train_loss = None, None, None
            cleanup()
            print('GPU out of memory! Performing garbage collection.')
            sys.stdout.flush()
            continue
        if i % report_every == 0 and i > 0: # report average loss at regular intervals
            print(f'step {i} (train): {np.mean(train_losses[-report_every:])}')
            sys.stdout.flush()
        if i % validate_every == 0:
            print("Validating on a sample...")
            sys.stdout.flush()
            avg_chrf = 0
            num_val_lang = math.ceil(len(dev_bitext_list)/3)
            for _ in range(num_val_lang):
                bitext_index = random.choices(range(len(dev_bitext_list)), weights=sampling_probs, k=1)[0]
                dev_bitext = dev_bitext_list[bitext_index]
                if dev_bitext.lang1_code == "eng_Latn" or dev_bitext.lang2_code == "eng_Latn":
                    src_texts = []
                    tgt_texts = []
                    for _ in range(100):
                        pair = next(iter(dev_bitext))
                        src_texts.append(pair[0])
                        tgt_texts.append(pair[1])
                        assert len(src_texts) == len(tgt_texts)
                else:
                    src_texts, tgt_texts = dev_bitext.lang1_sents, dev_bitext.lang2_sents
                candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=dev_bitext.lang1_code, tgt_lang=dev_bitext.lang2_code)
                for candidate, gold in zip(candidate_translations[:5], tgt_texts[:5]):
                    print('-'*5)
                    print(f'candidate: {candidate}')
                    print(f'gold:      {gold}')
                    sys.stdout.flush()
                bleu, chrf = evaluate_translations(candidate_translations, tgt_texts)
                avg_chrf += chrf
            avg_chrf /= num_val_lang
            
            # only save model if there is an imporvement on chrf score after certain number of epochs
            # TODO: see if I want this do depend on both loss and chrf
            print(f"Average chrF: {avg_chrf}")
            if avg_chrf > last_best_chrf or i <= (patience): 
                print("Saving new best model!")
                sys.stdout.flush()
                tokenizer.save_pretrained(finetuned_model_dir) 
                model.save_pretrained(finetuned_model_dir)
                last_saved_model = model
                last_saved_tokenizer = tokenizer
                last_best = i
                last_best_chrf = avg_chrf 
               
        if i - last_best >= patience:
            print('No imporvement in ', patience, ' epochs. Stopping training.' )
            sys.stdout.flush()
            break
        
    return last_saved_model, last_saved_tokenizer


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
    lps = NLLB_SEED_LPS if args.data == 'nllb-seed' else AMERICAS_NLP_LPS 
    if len(lps) > 1:
        train_scope = 'multi'
    else:
        train_scope = 'bi'
    corpus = MultilingualCorpus(csv_file)
    train_data = corpus.create_mixture_of_bitexts(lps, batch_size=2, split = 'train')
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
            if eval_bitext.lang1_code == "eng_Latn" or eval_bitext.lang2_code == "eng_Latn": # handel streaming bitexts
                eval_bitext = corpus.create_bitext('spa_Latn', 'eng_Latn', 'dev', lang1_file = 'opus_data/dev.es', lang2_file = 'opus_data/dev.es')
                src_texts = []
                tgt_texts = []
                for _ in range(100):
                    pair = next(iter(eval_bitext))
                    src_texts.append(pair[0])
                    tgt_texts.append(pair[1])
                    assert len(src_texts) == len(tgt_texts)
            else:
                eval_bitext = corpus.create_bitext(src, tgt, 'dev')
                src_texts, tgt_texts = eval_bitext.lang1_sents, eval_bitext.lang2_sents
                
            candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=eval_bitext.lang1_code, tgt_lang=eval_bitext.lang2_code)
            bleu, chrf = evaluate_translations(candidate_translations, tgt_texts)
            tgt_lang = AMERICAS_NLP_CODE_TO_LANG[tgt.split('_')[0]]
            log_evaluation(LOG_FILE, model_dir.replace("models/", ""), train_scope, tgt_lang, bleu, chrf, notes) # log a line for each evaluation
            print('Wrote all evaluations to ', LOG_FILE)
        except Exception as e:
            print(f"Error occurred while evaluating {src} --> {tgt}: {str(e)}")
        
    