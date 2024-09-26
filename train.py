import pandas as pd
import sys
from transformers.optimization import Adafactor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_constant_schedule_with_warmup
from preprocess import preproc
import random
import gc
import torch
from tqdm import tqdm
import numpy as np

model_name = "facebook/nllb-200-distilled-600M"
tsv_file = '/mnt/storage/hopkins/thesis/data/rus_tyv_parallel_50k.tsv'
batch_size = 16  # 32 already doesn't fit well to 15GB of GPU memory
max_length = 128  # token sequences will be truncated
training_steps = 60000  
train_losses = []  # with this list, I do very simple tracking of average loss
dev_losses = []  # with this list, I do very simple tracking of average loss
MODEL_SAVE_PATH = '/mnt/storage/hopkins/models/nllb-rus-tyv-v3' 


trans_df = pd.read_csv(tsv_file, sep="\t")
df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
df_test = trans_df[trans_df.split=='test'].copy()   # 500 items

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
LANGS = [('ru', 'rus_Cyrl'), ('tyv', 'kir_Cyrl')]

def get_batch_pairs(batch_size, data=df_train):
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



x, y, train_loss = None, None, None
x_dev, y_dev, dev_loss = None, None, None
best_dev_loss = None
cleanup()

for i in tqdm(range(len(train_losses), training_steps)):
    xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
    xx_dev, yy_dev, lang1_dev, lang2_dev = get_batch_pairs(batch_size, data=df_dev)

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
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        best_dev_loss = dev_loss