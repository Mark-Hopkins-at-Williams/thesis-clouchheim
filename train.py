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

# source define language and model
SRC_LANG = ('spanish', 'es', 'spa_Latn') # full name, abbreviation, nllb tag
BASE_MODEL = "facebook/nllb-200-distilled-600M"

########## SET TRAINING MODE ##############
MODE = 'multi' # set to multi or mono to change what this trains on

model, tokenizer = load_model_untrained(BASE_MODEL)
print(f'loaded tokenizer and model ({BASE_MODEL}) to finetune')

if MODE == 'multi':
    MODEL_SAVE_PATH = f"/mnt/storage/clouchheim/models/nllb_multilingual_v2"
    
    df_train, df_dev = load_all_data()
    print(f'Loaded and created {SRC_LANG[0]} to ALL LANGS data')
    
    model, tokenizer = add_all_langs_to_tokenizer(model, tokenizer, MODEL_SAVE_PATH) 
    lang_tags = ['cni_Latn',
                'aym_Latn',
                'bzd_Latn',
                'ctp_Latn',
                'oto_Latn',
                'nah_Latn',
                'tar_Latn',
                'shp_Latn', 
                'hch_Latn',
                'grn_Latn',
                'quy_Latn']
    
else: #mono
    
    ########### DEFINE MONO TARGET LANGUAGE ###############
    TGT_LANG = ('nahuatl', 'nah', 'nah_Latn')
    MODEL_SAVE_PATH = f"/mnt/storage/clouchheim/models/nllb_{SRC_LANG[0]}_{TGT_LANG[0]}"
    
    df_train, df_dev = load_data(SRC_LANG[0], SRC_LANG[1], TGT_LANG[0], TGT_LANG[1], TGT_LANG[2])
    print(f'Loaded and created {SRC_LANG[0]} to {TGT_LANG[0]} data')
    
    model, tokenizer = add_language_tag_to_tokenizer(model, tokenizer, TGT_LANG[2], MODEL_SAVE_PATH)
    # update model size
    similar_lang_tag = 'grn_Latn'
    model = update_model_for_new_token(model, tokenizer, similar_lang_tag, new_lang_tag)
    
    lang_tags = [TGT_LANG[2]] # just the single target langugae
    
# train model
print('starting training')
train_model(model, tokenizer, df_train, df_dev, SRC_LANG[2], lang_tags, MODEL_SAVE_PATH)


print(f'done training, saved to {MODEL_SAVE_PATH}')
