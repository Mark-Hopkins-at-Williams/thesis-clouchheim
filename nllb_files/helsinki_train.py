import gc
import os
import sys
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from finetune import tokenize
from validate import evaluate_translations, log_evaluation, batched_translate
from finetune import finetune
from multilingualdata import MultilingualCorpus, MixtureOfBitexts, StreamingBitext, Bitext
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from configure import ALL_AMERICAS_NLP_LANGS, USE_CUDA
import shutil  # for deleting directories

def finetune_n_best(mixture_of_bitexts, dev_bitext_list, base_model, finetuned_model_dir,
                  training_steps=60000,
                  max_length=128,  # token sequences will be truncated to this many tokens
                  report_every=100,
                  validate_every=1000, 
                  sampling_probs=None,
                  num_best_save=1
                  ):
    
    # set up sampling probs for dev
    if sampling_probs:
        sampling_probs = [p / sum(sampling_probs) for p in sampling_probs]
    else:
        sampling_probs = [1 / len(mixture_of_bitexts.bitexts)] * len(mixture_of_bitexts.bitexts)
    
    # Load model and tokenizer
    if isinstance(base_model, list):  # handle pretrained model input as list
        model = base_model[0]
        tokenizer = base_model[1]
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
    new_lang_codes = [code for code in mixture_of_bitexts.get_language_codes() if code in tokenizer.get_vocab()]
    tokenizer.add_tokens(new_lang_codes)
    model.resize_token_embeddings(len(tokenizer))
    if USE_CUDA:
        model.cuda()
    
    # Set up optimizer and scheduler
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

    # Initialize training
    last_best = 0
    patience = 80000
    train_losses = []
    best_models = []  # List to store best models and their scores

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
        except RuntimeError:
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            print('GPU out of memory! Performing garbage collection.')
            sys.stdout.flush()
            continue
        
        # Reporting average loss
        if i % report_every == 0 and i > 0:
            print(f'step {i} (train): {np.mean(train_losses[-report_every:])}')
            sys.stdout.flush()
        # Validation and saving models
        if i % validate_every == 0:
            print("Validating on a sample...")
            sys.stdout.flush()
            avg_chrf = 0
            num_val_lang = math.ceil(len(dev_bitext_list) / 3)
            print("# evals at this validation:", num_val_lang)
            sys.stdout.flush()
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
                bleu, chrf = evaluate_translations(candidate_translations, tgt_texts)
                avg_chrf += chrf
            avg_chrf /= num_val_lang
            print(f"Avg chrF = {avg_chrf}")
            sys.stdout.flush()
            # Check if the current model is one of the top models
            if len(best_models) < num_best_save or avg_chrf > min(best_models, key=lambda x: x[0])[0]:
                if num_best_save == 1:
                    model_dir = finetuned_model_dir
                else:
                    model_dir = f"{finetuned_model_dir}/model_step_{i}"
                print(f"Saving new best model to {model_dir}!")
                sys.stdout.flush()
                tokenizer.save_pretrained(model_dir) 
                model.save_pretrained(model_dir)
                
                # Add the new model and its score
                best_models.append((avg_chrf, model_dir))
                
                # Keep only the top `num_best_save` models
                if len(best_models) > num_best_save:
                    # Remove the model with the lowest score
                    worst_model = min(best_models, key=lambda x: x[0])
                    shutil.rmtree(worst_model[1])  # Delete the model directory
                    print(f"Removing {worst_model}")
                    sys.stdout.flush()
                    best_models.remove(worst_model)

            # Update last best tracking
            if avg_chrf > last_best:
                last_best = i

        # Check for patience
        if i - last_best >= patience:
            print('No improvement in', patience, 'epochs. Stopping training.')
            sys.stdout.flush()
            break
    
    # Return the directories of the saved models
    best_model_dirs = [model_dir for _, model_dir in sorted(best_models, reverse=True)]
    return best_model_dirs
    
def make_sampling_weights(eng_p, anlp_p):
    phase1_p = [eng_p] + [(anlp_p)/11] * 11
    return phase1_p

def helsinki_bitexts_list(opus_corpus, opus_files, anlp_corpus, split):
    bitext_list = []
    
    # append eng_spa
    bitext_list.append(opus_corpus.create_bitext('eng_Latn', 'spa_Latn', split, lang1_file = opus_files[0], lang2_file = opus_files[1]))
    # append all anlp _ spa
    for lang in ALL_AMERICAS_NLP_LANGS:
        bitext_list.append(anlp_corpus.create_bitext('spa_Latn', lang, split))
    return bitext_list

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Three-phase training for multilingual translation")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training")
    args = parser.parse_args()
    
    save_dir = args.save_dir
    print(f"Starting Training Script for {save_dir}")
    sys.stdout.flush()
    
    # Load in model and tokenizer from scratch
    print("Loading in model for training")
    sys.stdout.flush()
    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    config = AutoConfig.from_pretrained(base_model)
    model_base = AutoModelForSeq2SeqLM.from_config(config)
    
    # load in data 
    anlp_csv = "americas_nlp_data.csv"
    opus_csv = None
    opus_dir = "opus_data"
    opus_files_train = (f'{opus_dir}/train.en', f'{opus_dir}/train.es')
    opus_files_dev = (f'{opus_dir}/dev.en', f'{opus_dir}/dev.es')
    
    print("Loading in data")
    sys.stdout.flush()
    opus_corpus = MultilingualCorpus(opus_csv, streaming = True)
    anlp_corpus = MultilingualCorpus(anlp_csv) 
    
    train_bitexts = helsinki_bitexts_list(opus_corpus, opus_files_train, anlp_corpus, 'train')
    dev_bitexts = helsinki_bitexts_list(opus_corpus, opus_files_dev, anlp_corpus, 'dev')
    # create all bitexts
    
    # create phase 1 data 91% eng 9% indigenous
    print('Making phase 1 data')
    sys.stdout.flush()
    phase1_p = make_sampling_weights(0.91, 0.09)
    phase1_data = MixtureOfBitexts(train_bitexts, batch_size = 2, sampling_probs = phase1_p)
    phase1_dev_data = [opus_corpus.create_bitext('eng_Latn', 'spa_Latn', 'dev', lang1_file = opus_files_dev[0], lang2_file = opus_files_dev[1])]
    phase1_dir = save_dir + 'phase_1'
        
    # train phase 1 
    print("Starting phase 1 training")
    sys.stdout.flush()
    phase1_model_tok = [model_base, tokenizer]
    phase1_model, phase1_tokenizer = finetune(phase1_data, phase1_dev_data, phase1_model_tok, phase1_dir, training_steps=100000)
    print(f'Done training phase 1, saved to {phase1_dir}. Now making phase 2 data')
    sys.stdout.flush()
        
    # create phase 2 data 37% eng 63% indegenous
    phase2_p = make_sampling_weights(0.37, 0.63)
    phase2_data = MixtureOfBitexts(train_bitexts, batch_size = 2, sampling_probs = phase2_p)
    phase2_dev_data = dev_bitexts
    phase2_dir = save_dir + 'phase_2'
    
    # train phase 2
    print("Starting phase 2 training")
    sys.stdout.flush()
    phase2_model_tok = [phase1_model, phase1_tokenizer]
    phase2_models = finetune_n_best(phase2_data, phase2_dev_data, phase2_model_tok, phase2_dir, training_steps=200000, sampling_probs = phase2_p, num_best_save = 5)
    print(f"best models from phase 2: {phase2_models}")
    sys.stdout.flush()
    