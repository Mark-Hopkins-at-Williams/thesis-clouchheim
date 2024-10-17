import gc
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from validate import evaluate, batched_translate
from configure import USE_CUDA
from configure import AMERICAS_NLP_CSV, AMERICAS_NLP_LPS
from configure import NLLB_SEED_CSV, NLLB_SEED_LPS
from multilingualdata import MultilingualCorpus


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


def finetune(mixture_of_bitexts, dev_bitext, base_model, finetuned_model_dir,
             training_steps=60000,
             max_length=128, # token sequences will be truncated to this many tokens
             report_every=100,
             validate_every=1000,
             ):    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
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
    patience = 30000
    cleanup()
    train_losses = []   # tracks average loss
    for i in tqdm(range(training_steps)):
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
            continue
        if i % report_every == 0 and i > 0: # report average loss at regular intervals         
            print(f'step {i} (train): {np.mean(train_losses[-report_every:])}')
            sys.stdout.flush()
        if i % validate_every == 0:
            print("Validating on a sample...")
            src_texts, tgt_texts = dev_bitext.lang1_sents, dev_bitext.lang2_sents
            candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=dev_bitext.lang1_code, tgt_lang=dev_bitext.lang2_code)
            for candidate, gold in zip(candidate_translations[:5], tgt_texts[:5]):
                print('-'*5)
                print(f'candidate: {candidate}')
                print(f'gold:      {gold}')
            evaluate(candidate_translations, tgt_texts)
            print("Saving new best model!")
            #TODO: save only if the evaluation result is better than previous
            tokenizer.save_pretrained(finetuned_model_dir) #TODO: check that we can use the same directory
            model.save_pretrained(finetuned_model_dir)           
            last_best = i        
        if i - last_best >= patience:
            break
 
 
def finetune_dress_rehearsal(mixture_of_bitexts, dev_bitext, base_model, finetuned_model_dir,
             training_steps=60000,
             max_length=128, # token sequences will be truncated to this many tokens
             report_every=100,
             validate_every=1000,
             ):    
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    new_tokens = ['nah_Latn', 'oto_Latn']
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    for i in tqdm(range(training_steps)):
        lang1_sents, lang2_sents, lang1, lang2 = mixture_of_bitexts.next_batch()  
        x = tokenize(lang1_sents, lang1, tokenizer, max_length)
        
        y = tokenize(lang2_sents, lang2, tokenizer, max_length, alt_pad_token=-100)
        print(y)
            
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning script for NLLB models.")
    parser.add_argument("--data", type=str, required=True, choices=['nllb-seed', 'americas-nlp'], help="Finetuning data")
    parser.add_argument("--model_dir", type=str, help="Directory for storing the trained model")
    parser.add_argument("--nllb_model", type=str, default="600M", choices=['600M', '1.3B', '3.3B'], help="NLLB base model.")
    parser.add_argument("--dev_src", type=str, required=True, help="Source language for validation.")
    parser.add_argument("--dev_tgt", type=str, required=True, help="Target language for validation.")
    
    args = parser.parse_args()
    model_dir = args.model_dir
    model_name = "facebook/nllb-200-distilled-" + args.nllb_model
    if os.path.exists(model_dir):
        print(f"model directory already exists: {model_dir}")
        exit()
    csv_file = NLLB_SEED_CSV if args.data == 'nllb-seed' else AMERICAS_NLP_CSV
    lps = NLLB_SEED_LPS if args.data == 'nllb-seed' else AMERICAS_NLP_LPS
    corpus = MultilingualCorpus(csv_file)
    train_data = corpus.create_mixture_of_bitexts(lps, batch_size=2)
    dev_bitext = corpus.create_bitext(args.dev_src, args.dev_tgt, 'dev')
    finetune(train_data, dev_bitext, model_name, model_dir)
    