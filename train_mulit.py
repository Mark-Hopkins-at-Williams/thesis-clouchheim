import gc
import os
import sys
import torch
import random
import evaluate
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from preprocess import preproc
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#from americasnlp import AMERICASNLP_CODES

# TODO
# add -e command line option to evaluate at the end


size = "600M"       #default size
batch_size = 16     # 32 already doesn't fit well to 15GB of GPU memory
max_length = 128    # token sequences will be truncated
training_steps = 60000  
model_save_path = '/mnt/storage/clouchheim/models/' 
csv_train = '/mnt/storage/clouchheim/thesis-clouchheim/data/train_spa_xx.csv'
csv_dev = '/mnt/storage/clouchheim/thesis-clouchheim/data/dev_spa_xx.csv'
dev_losses = []     
train_losses = []   # these lists track of average loss
src_lang = ""
evaluate = False    # do not run evaluate after finishing training

TGT_TAGS = ['cni_Latn',
            'ayr_Latn',
            'bzd_Latn',
            'ctp_Latn',
            'grn_Latn',
            'oto_Latn',
            'nah_Latn',
            'quy_Latn',
            'tar_Latn',
            'shp_Latn',
            'hch_Latn']

def get_batch_pairs(batch_size, data):
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    return xx, yy, long1, long2

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    
    my_model = 'nllb_spanish_multi_1016'
    print('Training: ', my_model)
    
    model_name = "facebook/nllb-200-distilled-600M"
    model_save_path += model_name
    
    now = datetime.now()
    # mm/dd/YY H:M:S
    date_time = now.strftime("%m/%d/%Y %H:%M:%S")

    df_train = pd.read_csv(csv_train, sep=",")
    df_dev = pd.read_csv(csv_dev, sep=",")  
    
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
    x_dev, y_dev, dev_loss = None, None, None
    best_dev_loss = None
    last_best = 0
    patience = 30000
    cleanup()
    
    
    subsets = {}
    
    # forming langueage subsets
    for tag in TGT_TAGS:
        train = df_train[df_train['lang'] == tag]
        dev = df_dev[df_dev['lang'] == tag]
        subsets[tag] = (train, dev)

    for i in tqdm(range(len(train_losses), training_steps)):
        
        #choose a tag, filter data to correct language pair
        tgt_lang_tag = random.choice(TGT_TAGS)
        subset_train, subset_dev = subsets[tgt_lang_tag] 
        LANGS = [('src', 'spa_Latn'), ('tgt', tgt_lang_tag)]
    
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size, subset_train)
        xx_dev, yy_dev, lang1_dev, lang2_dev = get_batch_pairs(batch_size, data=subset_dev)

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
            print(f'step {i} (train): {np.mean(train_losses[-1000:])}')
            print(f'step {i} (dev):   {np.mean(dev_losses[-1000:])}')
            sys.stdout.flush()

        if i % 1000 == 0 and i > 0 and (best_dev_loss is None or dev_loss < best_dev_loss):
            print("Saving new best model!")
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            best_dev_loss = dev_loss
            last_best = i
        
        if i - last_best >= patience:
            break