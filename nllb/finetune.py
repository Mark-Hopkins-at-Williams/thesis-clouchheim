import gc
import json
import os
import sys
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from configure import USE_CUDA
import matplotlib.pyplot as plt
from transformers.optimization import Adafactor
from multilingualdata import MultilingualCorpus
from transformers import get_constant_schedule_with_warmup
from encrypt import create_token_permuter, encrypt_sentences
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

def add_lines(sents, lang, current_max_id, start, end, split, data):
    for line in range(start, end + 1): #TODO; CHANGE THIS SO THAT I AM ADDING LINES RATHER THAN RELYING ON PERFECTLY PARALLEL APPENDS
        data['language'].append(lang)
        data['script'].append('Latn')
        data['sent_id'].append(line + current_max_id)
        data['text'].append(sents[line])
        data['split'].append(split)          
    return data         

def finetune(mixture_of_bitexts, dev_bitexts, base_model, finetuned_model_dir, training_steps,
             max_length=128, # token sequences will be truncated to this many tokens
             report_every=500,
             validate_every=500,
             patience=5,
             gpu_memory_fraction=None
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
        if gpu_memory_fraction:
            # Limit GPU memory usage to the specified fraction
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
        # Enable dynamic memory growth
        torch.cuda.set_device(0)  # Assumes a single GPU is being used
        torch.backends.cudnn.benchmark = True
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
   
def cache_relevant_lines(config):
    def in_any_interval(i, intervals):
        for (start, end) in intervals:
            if start <= i < end:
                return True
        return False

    necessary_lines = dict()  # keys are corpus_name, values are intervals
    for split in ["training_data", "validation_data", "test_data"]:
        for corpus_description in config[split]:
            corpus_id = corpus_description['corpus']
            start_index = corpus_description['start_index']
            end_index = corpus_description['end_index']
            if corpus_id not in necessary_lines:
                necessary_lines[corpus_id] = []
            necessary_lines[corpus_id].append((start_index, end_index)) #TODO: smoooosh intervals maybe

    cached_lines = dict()  # keys are (corpus_name, src/tgt, line_num), values are single sentences
    for corpus_name in necessary_lines:
        for lang in ['src_file', 'tgt_file']:
            filename = config['corpora'][corpus_name][lang]
            with open(filename) as reader:
                for i, line in enumerate(reader):  #TODO: reads through entire file, perhaps pre-compute max line num for early stopping
                    if in_any_interval(i, necessary_lines[corpus_name]):
                        cached_lines[(corpus_name, lang[:3], i)] = line     
    return cached_lines


def main():
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
    
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    
    #TODO: add the ability to have source permutations
    #TODO: add the ability to have multiple sources
    
    # get all relevent lines
    cached_lines = cache_relevant_lines(config) 
    
    # dispatch to token permuters
    permutation_metadata = dict()   # keys are the permutation indices, values is a list of dicts describing the data
    src_permutation_metadata = dict()
    for split in ["training_data", "validation_data", "test_data"]:
        for corpus_description in config[split]:
            if 'tgt_permutation' in corpus_description:
                perm_index = corpus_description['tgt_permutation']
                if perm_index not in permutation_metadata:
                    permutation_metadata[perm_index] = []
                permutation_metadata[perm_index].append(
                    {k: corpus_description[k] for k in corpus_description if k != 'tgt_permutation'}
                )
            if 'src_permutation' in corpus_description:
                perm_index = corpus_description['src_permutation']
                if perm_index not in src_permutation_metadata:
                    src_permutation_metadata[perm_index] = []
                src_permutation_metadata[perm_index].append(
                    {k: corpus_description[k] for k in corpus_description if k != 'src_permutation'}
                )
    
    # create permuters for each permutation given permutation metadata
    permuters = dict() # keys are permutation ids, vals are permuters
    for permuter_id in permutation_metadata:
        sents = []
        for metadata in permutation_metadata[permuter_id]:
            corpus_name = metadata['corpus']
            start_index = metadata['start_index']
            end_index = metadata['end_index']
            for sent_id in range(start_index, end_index):
                sent = cached_lines[(corpus_name, 'tgt', sent_id)]
                sents.append(sent)
        permuters[permuter_id] = create_token_permuter(tokenizer, sents) # TODO: *this is where the other tokenization of the same sentences is happening
    
    src_permuters = dict() # keys are permutation ids, vals are permuters
    for permuter_id in src_permutation_metadata:
        sents = []
        for metadata in src_permutation_metadata[permuter_id]:
            corpus_name = metadata['corpus']
            start_index = metadata['start_index']
            end_index = metadata['end_index']
            for sent_id in range(start_index, end_index):
                sent = cached_lines[(corpus_name, 'src', sent_id)]
                sents.append(sent)
        src_permuters[permuter_id] = create_token_permuter(tokenizer, sents)  
    
    # read and encrypt data, place into dataframe   
    data = []
    lps = []
    split_mapping = { "training_data": "train", "validation_data": "dev", "test_data": "test"}
    for permuter_id in permuters:
        permuter = permuters[permuter_id] 
        for split_data in ["training_data", "validation_data", "test_data"]:
            for metadata in config[split_data]:
                if metadata['tgt_permutation'] == permuter_id:
                    src_permute = False
                    if 'src_permutation' in metadata:
                        src_permute = True
                        src_permuter = src_permuters[metadata['src_permutation']]
                        print('found source permuter:', metadata['src_permutation'])
                    corpus_name = metadata['corpus']
                    if src_permute:
                        src_lang = corpus_name.split('-')[0] + str(metadata['src_permutation'])
                    else:
                        src_lang = corpus_name.split('-')[0]
                    tgt_lang = corpus_name.split('-')[1] + str(permuter_id)
                    lps.append([src_lang + '_Latn', tgt_lang + '_Latn'])
                    start_index = metadata['start_index']
                    end_index = metadata['end_index']
                    split = split_mapping[split_data] 
                    for sent_id in range(start_index, end_index):
                        # add target sents
                        tgt_sent = cached_lines[(corpus_name, 'tgt', sent_id)]
                        encrypted_tgt = encrypt_sentences(tgt_sent, tokenizer, permuter) # TODO: this has a redundant tokenization* (check encrypt.py), but to maintian sent_id I couldnt find another option
                        data.append({'language': tgt_lang, 'script': 'Latn', 'sent_id': sent_id, 'text': encrypted_tgt[0], 'split': split})
                        # add source sents
                        src_sent = cached_lines[(corpus_name, 'src', sent_id)]
                        if src_permute:
                            src_sent = encrypt_sentences(src_sent, tokenizer, src_permuter)
                            src_sent = src_sent[0]
                        data.append({'language': src_lang, 'script': 'Latn', 'sent_id': sent_id, 'text': src_sent.strip(), 'split': split})
                
    # convert into pandas dataframe and create mixture of bitexts   
    bitexts = pd.DataFrame(data)
    bitexts = bitexts.drop_duplicates(subset=["language", "sent_id"])
    lps = list(set(tuple(pair) for pair in lps))
    print('language pairs in model:', lps)
    
    #bitexts.to_csv(model_dir +'/data.csv', index=False)
    
    corpus = MultilingualCorpus(bitexts) 
    train_data = corpus.create_mixture_of_bitexts(lps, batch_size=32, split='train')
    dev_data = corpus.create_mixture_of_bitexts(lps, batch_size=32, split='dev')
    model_name = config['base_model']
    
    # finetune model 
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
        with open(os.path.join(model_dir, f'source.{lp[0]}'), 'w') as writer:
            for src in src_texts:
                writer.write(f'{src}\n')
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


if __name__ == "__main__":
    main()