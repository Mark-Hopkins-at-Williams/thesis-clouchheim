import sacrebleu
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer
from general_nllb_functions import *

####################### Set Variables ######################
MODEL_LOAD_NAME = '/mnt/storage/clouchheim/models/nllb_multilingual'  # this model name is swapped
#MODEL_LOAD_NAME = 'facebook/nllb-200-distilled-600M' # not fine tuned version
SRC_LANG = ('spanish', 'es', 'spa_Latn')
TGT_LANG = ('bribri', 'bzd', 'bzd_Latn')
LANGS = [SRC_LANG, TGT_LANG]

######################## EVALUATE MODEL ###########################
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_LOAD_NAME, local_files_only=True).cuda()
tokenizer = NllbTokenizer.from_pretrained(MODEL_LOAD_NAME)
print('loaded model', MODEL_LOAD_NAME)

# Get Metrics
bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

# Load Data
print('starting translation from ', SRC_LANG[0], ' to ', TGT_LANG[0])
df_train, df_dev = load_data(SRC_LANG[0], SRC_LANG[1], TGT_LANG[0], TGT_LANG[1])
translations = batched_translate(df_dev['src'].tolist(), SRC_LANG[2], TGT_LANG[2], model, tokenizer)
print('finished translation!')

# print evaluations 
print('Model Evaluation:')
print(bleu_calc.corpus_score(translations, [df_dev['tgt'].tolist()]))
print(chrf_calc.corpus_score(translations, [df_dev['tgt'].tolist()]))