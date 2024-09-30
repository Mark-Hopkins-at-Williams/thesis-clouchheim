import pandas as pd
import sys
from transformers.optimization import Adafactor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_constant_schedule_with_warmup 
import random
import gc
import torch
from tqdm import tqdm
import numpy as np
from general_nllb_functions import *

########## Main method to load data, add a language tag, and then train a model ##############

# define languages
SRC_LANG = ('spanish', 'es', 'spa_Latn') # full name, abbreviation, nllb tag
TGT_LANG = ('bribri', 'bzd', 'bzd_Latn') 
BASE_MODEL = "facebook/nllb-200-distilled-600M"
MODEL_SAVE_PATH = f"/mnt/storage/clouchheim/models/nllb_{SRC_LANG[0]}_{TGT_LANG[0]}_tagtest"
    
# load data
df_train, df_dev = load_data(SRC_LANG[0], SRC_LANG[1], TGT_LANG[0], TGT_LANG[1])
print(f'Loaded and created {SRC_LANG[0]} to {TGT_LANG[0]} data')
    
# load model and tokenizer
model, tokenizer = load_model_untrained(BASE_MODEL)
print(f'loaded tokenizer and model ({BASE_MODEL}) to finetune')

tokenizer = add_language_tag_to_tokenizer(tokenizer, TGT_LANG[2], MODEL_SAVE_PATH)
model = update_model_for_new_token(model, tokenizer)

# train model
print('starting training')
train_model_2(model, tokenizer, df_train, df_dev, SRC_LANG[2], TGT_LANG[2], MODEL_SAVE_PATH, training_steps = 2000)
print(f'done training, saved to {MODEL_SAVE_PATH}')
