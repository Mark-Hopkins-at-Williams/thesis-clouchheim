import gc
import json
import matplotlib.pyplot as plt
import os
import shutil
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from configure import USE_CUDA
from multilingualdata import MultilingualCorpus
from validate import evaluate_translations, batched_translate

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


def finetune(mixture_of_bitexts, dev_bitexts, base_model, finetuned_model_dir, training_steps,
             max_length=128, # token sequences will be truncated to this many tokens
             report_every=500,
             validate_every=500,
             patience=10
             ):    
    print('Training', finetuned_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    new_lang_codes = [code for code in mixture_of_bitexts.get_language_codes() if code not in tokenizer.get_vocab()]
    print(f"Augmenting vocabulary with the following tokens:")
    for lang_code in new_lang_codes:
        print(f"  {lang_code}")
    tokenizer.add_special_tokens({"additional_special_tokens": new_lang_codes})
    tokenizer.save_pretrained(finetuned_model_dir)
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
    best_dev_loss = None
    max_patience = patience
    cleanup()
    train_losses = []   # tracks average loss
    train_plot_x, train_plot_y = [], []
    dev_plot_x, dev_plot_y = [], []
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
            train_plot_x.append(i)
            train_plot_y.append(np.mean(train_losses[-report_every:]))
            sys.stdout.flush()
        if i % validate_every == 0:
            print("Validating on a sample...")   
            model.eval()         
            dev_losses = []
            dev_batches = 100
            for _ in range(dev_batches):
                lang1_sents, lang2_sents, lang1, lang2 = dev_bitexts.next_batch()                
                x = tokenize(lang1_sents, lang1, tokenizer, max_length).to(model.device)
                y = tokenize(lang2_sents, lang2, tokenizer, max_length, alt_pad_token=-100).to(model.device)
                with torch.no_grad():
                    dev_loss = model(**x, labels=y.input_ids).loss
                    dev_losses.append(dev_loss.item())
            dev_loss = np.mean(dev_losses)
            dev_plot_x.append(i)
            dev_plot_y.append(dev_loss)
            print(f'dev loss: {dev_loss}')
            sys.stdout.flush()
            # plot the current training progress
            plt.clf()
            plt.plot(train_plot_x, train_plot_y, label='train', color='blue', linewidth=2)  
            plt.plot(dev_plot_x, dev_plot_y, label='dev', color='red', linewidth=2) 
            plt.xlabel("training steps")
            plt.ylabel("loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(finetuned_model_dir, 'training.png'))
            
            # save only if the evaluation result is better than previous best
            if best_dev_loss is None or dev_loss < best_dev_loss:
                print("Saving new best model!")
                model.save_pretrained(finetuned_model_dir)  
                patience = max_patience    
                best_dev_loss = dev_loss
            else:                
                patience -= 1               
                print(f"Model is worse than the best so far. Current patience: {patience}") 
        if patience <= 0:
            break
   
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning script for NLLB models.")
    parser.add_argument("--config", type=str, required=True, help="Experiment configuration (JSON)")
    parser.add_argument("--model_dir", type=str, help="Directory for storing the trained model")
    parser.add_argument("--steps", type=int, default=60_000, help="Number of training steps")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = json.load(file)
    model_dir = args.model_dir        
    model_version = 0
    while os.path.exists(model_dir + f"-v{model_version}"):
        model_version += 1
    model_dir = model_dir + f"-v{model_version}"
    os.mkdir(model_dir)
    shutil.copyfile(args.config, os.path.join(model_dir, 'experiment.json'))       
    
    ###################### Change to work with new json ##########################
    
    train = config['training_data']
    dev = config['validation_data']
    test = config['test_data']
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    all_data = train + dev + test
    
    # get scope of data needed and read in for each individual copora
    indices_by_corpus = defaultdict(lambda: {"lowest_start_index": float('inf'), "highest_end_index": float('-inf')})

    # Process all entries and update the dictionary
    for entry in all_data:
        corpus = entry['corpus']
        indices_by_corpus[corpus]['lowest_start_index'] = min(indices_by_corpus[corpus]['lowest_start_index'], entry['start_index'])
        indices_by_corpus[corpus]['highest_end_index'] = max(indices_by_corpus[corpus]['highest_end_index'], entry['end_index'])

    # Add the corresponding file paths from corpora (one entry for each corpus)
    corpora_scope = {}
    for corpus, indices in indices_by_corpus.items():
        corpora_scope[corpus] = {
            "src_file": config['corpora'][corpus]['src_file'],
            "tgt_file": config['corpora'][corpus]['tgt_file'],
            "lowest_start_index": indices['lowest_start_index'],
            "highest_end_index": indices['highest_end_index']
        }
    
    data = {'language': [], 'script': [], 'sent_id': [], 'text': [], 'split': []} # data frame to add all bitexts to 
     
    # tokenize and get ids for all langauges
    tgt_sents = {}
    for corpus, info in copora_scope.items():
        tgt_file = info['tgt_file']
        start = info['lowest_start_index']
        end = info['highest_end_index']
        sents = []
        with open(tgt_file, 'r') as reader:
            for current_index, line in enumerate(reader):  
                if start <= current_index <= end: 
                    sents.append(line.strip())
                elif current_index > end:  
                    break
                
        tgt_sents[corpus] = sents
        
        # tokenize and get ids
        tokenized = tokenizer(tgt_sents, return_tensors='pt', padding=True, truncation=True, max_length=128)
        ids = [idx for idx in tokenized['input_ids'].unique().tolist() if 4 <= idx <= 256000]
    
        # use create_token_permuter but wihtou repetition HEREHEHEHEHEHE
        
    
    # tokenize each sentence
    
    # for loop starts here
    # create a permuter for each aritifical langauge pair using the correct indicies
    
    # permute each language and add to a data frame with the language pairs, split, and sentneces
        # ex// 
        # language,script,sent_id,text,split
        # p0r,Latn,0,custasqpena bí irponsa,test
    
          
    csv_file = config['csv_file']
    lps = config['lps']
    corpus = MultilingualCorpus(csv_file) # change corpus to work with data frame rather than csv file
    train_data = corpus.create_mixture_of_bitexts(lps, batch_size=32, split='train')
    dev_data = corpus.create_mixture_of_bitexts(lps, batch_size=32, split='dev')
    model_name = config['base_model']
    ###################### 
    
    finetune(train_data, dev_data, model_name, model_dir, training_steps=args.steps)
    
    # now evaluate the trained model
    bleu_scores = []
    chrf_scores = []
    for lp in lps:
        bitext = corpus.create_bitext(lp[0], lp[1], 'test')
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        model.cuda()
        src_texts = bitext.lang1_sents
        tgt_texts = bitext.lang2_sents
        candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=lp[0], tgt_lang=lp[1])        
        bleu, chrf = evaluate_translations(candidate_translations, tgt_texts)
        with open(os.path.join(model_dir, f'candidates.{lp[1]}'), 'w') as writer:
            for candidate in candidate_translations:
                writer.write(f'{candidate}\n')
        with open(os.path.join(model_dir, f'references.{lp[1]}'), 'w') as writer:
            for ref in tgt_texts:
                writer.write(f'{ref}\n')
        bleu_scores.append(bleu)
        chrf_scores.append(chrf)
    with open(os.path.join(model_dir, 'scores.csv'), 'w') as writer:
        writer.write(','.join(['src', 'tgt', 'bleu', 'chrf']) + '\n')
        for i, (src, tgt) in enumerate(lps):
            writer.write(','.join([src, tgt, str(bleu_scores[i]), str(chrf_scores[i])]) + '\n')