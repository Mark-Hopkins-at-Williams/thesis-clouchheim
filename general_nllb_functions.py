import re
import unicodedata
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor, get_constant_schedule_with_warmup
import torch
import gc
import random
from sacremoses import MosesPunctNormalizer
import pandas as pd
import sys
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from tqdm.auto import tqdm, trange
import numpy as np
from os.path import join
import pandas as pd

################### Functions to Load Data ####################

def load_data(src_lang='spanish', src_lang_abbrv='es', tgt_lang='guarani', tgt_lang_abbrv='gn'):
    '''
    Function that given a source langauge, taget language, and the correct tags will 
    build the train and dev data frames for that language from AmericasNLP data.
    '''

    AMERICAS_NLP_DIR = '/mnt/storage/clouchheim/data/americasnlp2024/ST1_MachineTranslation/data/'
    SRC_LANG = (src_lang, src_lang_abbrv)
    TGT_LANG = (tgt_lang, tgt_lang_abbrv)

    # load in training data and create df_train
    with open(join(AMERICAS_NLP_DIR, f'{TGT_LANG[0]}-{SRC_LANG[0]}', f'train.{SRC_LANG[1]}'), 'r', encoding='utf-8') as f:
        src_sentences = f.readlines()
    with open(join(AMERICAS_NLP_DIR, f'{TGT_LANG[0]}-{SRC_LANG[0]}', f'train.{TGT_LANG[1]}'), 'r', encoding='utf-8') as f:
        tgt_sentences = f.readlines()
        
    src_sentences = [line.strip() for line in src_sentences]
    tgt_sentences = [line.strip() for line in tgt_sentences]
        
    # column names: ['id', 'src', 'tgt']
    df_train = pd.DataFrame({    
        'id': range(len(src_sentences)),
        'src': src_sentences,
        'tgt': tgt_sentences
    })

    # load dev data and create df_dev
    with open(join(AMERICAS_NLP_DIR, f'{TGT_LANG[0]}-{SRC_LANG[0]}', f'dev.{SRC_LANG[1]}'), 'r', encoding='utf-8') as f:
        src_sentences_dev = f.readlines()
    with open(join(AMERICAS_NLP_DIR, f'{TGT_LANG[0]}-{SRC_LANG[0]}', f'dev.{TGT_LANG[1]}'), 'r', encoding='utf-8') as f:
        tgt_sentences_dev = f.readlines()

        
    src_sentences_dev = [line.strip() for line in src_sentences_dev]
    tgt_sentences_dev = [line.strip() for line in tgt_sentences_dev]
        
    # column names: ['id', 'src', 'tgt']
    df_dev = pd.DataFrame({    
        'id': range(len(src_sentences_dev)),
        'src': src_sentences_dev,
        'tgt': tgt_sentences_dev
    })
    return df_train, df_dev

################### Functions to Pre Process and Train ####################

def get_non_printing_char_replacer(replace_by: str = " "):
    '''Return functon to replace certain characters with given'''
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

def preproc(text):
    '''Normalize text both font and characters'''
    mpn = MosesPunctNormalizer(lang="en")
    mpn.substitutions = [
        (re.compile(r), sub) for r, sub in mpn.substitutions
    ]
    replace_nonprint = get_non_printing_char_replacer(" ")
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean

def load_model_untrained(model_name = "facebook/nllb-200-distilled-600M"):
    '''Load model and tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def get_batch_pairs(batch_size, data, langs):
    '''get bathches of senteces of size batch_size for given languages'''
    (l1, long1), (l2, long2) = random.sample(langs, 2)
    indices = random.sample(range(len(data)), batch_size)
    xx = [preproc(data.iloc[i][l1]) for i in indices]
    yy = [preproc(data.iloc[i][l2]) for i in indices]
    return xx, yy, long1, long2
    
def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def add_language_tag_to_tokenizer(tokenizer, new_lang_tag, model_save_path):
    ''' Add the new language tag if it doesn't exist in the tokenizer '''
    if new_lang_tag not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [new_lang_tag]})
        tokenizer.save_pretrained(model_save_path)
        print(f"Tag '{new_lang_tag}' added successfully.")
    else:
        print(f"Tag '{new_lang_tag}' already exists in the tokenizer.")
        
    if new_lang_tag not in tokenizer.get_vocab():
        raise ValueError(f"Failed to add new language tag '{new_lang_tag}' to the tokenizer.")
    
    return tokenizer

def update_model_for_new_token(model, tokenizer):
    ''' Resize model embeddings to include the new token '''
    model.resize_token_embeddings(len(tokenizer))
    
    # Check if the model's vocabulary size was updated
    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        raise ValueError(f"Model embedding size not updated for new token '{new_lang_tag}'.")
    
    return model

def train_model_2(model, tokenizer, df_train, df_dev, src_lang_tag, tgt_lang_tag, model_save_path,
                batch_size = 16, max_length = 128, training_steps = 60000):
    model.cuda();
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
    model.train()
    x, y, loss = None, None, None
    cleanup()

    losses = []
    LANGS = [('src', src_lang_tag), ('tgt', tgt_lang_tag)]
    tq = trange(len(losses), training_steps)
    for i in tq:
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size, df_train, LANGS)
        try:
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            tokenizer.src_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            # -100 is a magic value ignored in the loss function
            # because we don't want the model to learn to predict padding ids
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        except RuntimeError as e:  # usually, it is out-of-memory
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            cleanup()
            print('error', max(len(s) for s in xx + yy), e)
            continue

        if i % 1000 == 0:
            # each 1000 steps, I report average loss at these steps
            print(i, np.mean(losses[-1000:]))

        if i % 1000 == 0 and i > 0:
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
        
        
def train_model(model, tokenizer, df_train, df_dev, src_lang_tag, tgt_lang_tag, model_save_path,
                batch_size = 16, max_length = 128, training_steps = 60000):
    '''Finetune model for given languages'''
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
    train_losses = []  # with this list, I do very simple tracking of average loss
    dev_losses = []  # with this list, I do very simple tracking of average loss

    LANGS = [('src', src_lang_tag), ('tgt', tgt_lang_tag)]
    
    ############# ADD A CHECK TO SEE IF THE TGT LANG HAS A VALID TAG (aka not Guarani) ##########
    ## the add tag function goes here

    x, y, train_loss = None, None, None
    x_dev, y_dev, dev_loss = None, None, None
    best_dev_loss = None
    cleanup()

    tq = trange(len(train_losses), training_steps)
    for i in tq:
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size, df_train, LANGS)
        xx_dev, yy_dev, lang1_dev, lang2_dev = get_batch_pairs(batch_size, df_dev, LANGS)

        try:
            model.train()
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            x_dev = tokenizer(xx_dev, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            tokenizer.src_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            y_dev = tokenizer(yy_dev, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            # -100 is a magic value ignored in the loss function
            # because we don't want the model to learn to predict padding ids
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
            y_dev.input_ids[y_dev.input_ids == tokenizer.pad_token_id] = -100

            train_loss = model(**x, labels=y.input_ids).loss
            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            with torch.no_grad():
                model.eval()
                dev_loss = model(**x_dev, labels=y_dev.input_ids).loss
                dev_losses.append(dev_loss.item())

        except RuntimeError as e:  # usually, it is out-of-memory
            optimizer.zero_grad(set_to_none=True)
            x, y, train_loss = None, None, None
            x_dev, y_dev, dev_loss = None, None, None
            cleanup()
            print('error', max(len(s) for s in xx + yy), e)
            continue

        if i % 1000 == 0:
            # each 1000 steps, I report average loss at these steps
            print(f'\nstep {i} (train): {np.mean(train_losses[-1000:])}')
            print(f'step {i} (dev):   {np.mean(dev_losses[-1000:])}')
            sys.stdout.flush()

        if i % 1000 == 0 and i > 0 and (best_dev_loss is None or dev_loss < best_dev_loss):
            print("Saving new best model!")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            best_dev_loss = dev_loss
            
        cleanup()

################### Functions to Evalaluate ####################

def translate(
    text, src_lang, tgt_lang, 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    """Translates a string or a list of strings."""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    model.eval() # turn off training mode
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
    """Translate texts in batches of similar length"""
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]