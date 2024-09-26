# Translate from Guarani to Spanish
# Fine tune nllb model in the same way as the tyvan russian tutorial

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_constant_schedule_with_warmup, NllbTokenizer
import re
from tqdm.auto import tqdm, trange
import random
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer
from transformers.optimization import Adafactor
import gc 
import torch
import sacrebleu 
import numpy as np
from datasets import load_dataset
from collections import Counter

################################ Load data ################################
#TODO: Load Data (train and dev)

# load in training data and create df_train
with open('/mnt/storage/clouchheim/data/americasnlp2024/ST1_MachineTranslation/data/guarani-spanish/train.es', 'r', encoding='utf-8') as f:
    es_sentences = f.readlines()
with open('/mnt/storage/clouchheim/data/americasnlp2024/ST1_MachineTranslation/data/guarani-spanish/train.gn', 'r', encoding='utf-8') as f:
    gn_sentences = f.readlines()
    
es_sentences = [line.strip() for line in es_sentences]
gn_sentences = [line.strip() for line in gn_sentences]
    
# column names: ['id', 'es', 'gn']
# shape: (26302, 3)
df_train = pd.DataFrame({    
    'id': range(len(es_sentences)),
    'es': es_sentences,
    'gn': gn_sentences
})

# load dev data and create df_dev
with open('/mnt/storage/clouchheim/data/americasnlp2024/ST1_MachineTranslation/data/guarani-spanish/dev.es', 'r', encoding='utf-8') as f:
    es_sentences_dev = f.readlines()
with open('/mnt/storage/clouchheim/data/americasnlp2024/ST1_MachineTranslation/data/guarani-spanish/dev.gn', 'r', encoding='utf-8') as f:
    gn_sentences_dev = f.readlines()
    
es_sentences_dev = [line.strip() for line in es_sentences_dev]
gn_sentences_dev = [line.strip() for line in gn_sentences_dev]
    
# column names: ['id', 'es', 'gn']
# shape: (995, 3)
df_dev = pd.DataFrame({    
    'id': range(len(es_sentences_dev)),
    'es': es_sentences_dev,
    'gn': gn_sentences_dev
})

##### Load model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

##### Test - spanish to english
tokenizer.src_lang = "spa_Latn"
'''inputs = tokenizer(text="Ella no tenía ni idea de dónde mirar.", return_tensors="pt")
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])   ########## I HAD TO CHANGE TO VERSION 4.33.0 TO USE THE LANG_CODE_TO_ID
print(tokenizer.decode(translated_tokens[0], skip_special_tokens=True))'''
# She had no idea where to look.

########################## Tokenize #########################
def word_tokenize(text):
    """
    Split a text into words, numbers, and punctuation marks
    (for languages where words are separated by spaces)
    """
    return re.findall('(\w+|[^\w\s])', text)

##### tokenize a sample to see number if the words per token matches somewhat
smpl = df_train.sample(10000, random_state=1)
smpl['es_toks'] = smpl.es.apply(tokenizer.tokenize)
smpl['gn_toks'] = smpl.gn.apply(tokenizer.tokenize)
smpl['es_words'] = smpl.es.apply(word_tokenize)
smpl['gn_words'] = smpl.gn.apply(word_tokenize)

stats = smpl[
    ['es_toks', 'gn_toks', 'es_words', 'gn_words']
].applymap(len).describe()
print("Spanish toks per word:", stats.es_toks['mean'] / stats.es_words['mean'])  # number of tokens per word Spanish: 1.330
print("Guarani toks per word:", stats.gn_toks['mean'] / stats.gn_words['mean'])  # number of tokens per word Guarani: 1.609

##### check for unknowns
texts_with_unk_gn = [
    text for text in tqdm(df_train.gn) 
    if tokenizer.unk_token_id in tokenizer(text).input_ids
]
print("Number of <unk> per 1000 examples of Guarani:", len(texts_with_unk_gn))
# Number of <unk> per 1000 examples of Guarani: 6408 

texts_with_unk_es = [
    text for text in tqdm(df_train.es) 
    if tokenizer.unk_token_id in tokenizer(text).input_ids
]
print("Number of <unk> per 1000 examples of Spanish:", len(texts_with_unk_es))
# Number of <unk> per 1000 examples of Spanish: 4719

##### preprocess and normalize
mpn = MosesPunctNormalizer(lang="es")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]

def get_non_printing_char_replacer(replace_by: str = " "):
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

replace_nonprint = get_non_printing_char_replacer(" ")

def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean

# check number of unknowns post pre-processing
texts_with_unk_normed_gn = [
    text for text in tqdm(texts_with_unk_gn) 
    if tokenizer.unk_token_id in tokenizer(preproc(text)).input_ids
]
print("# of <unk> after pre-processing Guarani:", len(texts_with_unk_normed_gn)) # 0 - unknown after pre-processing
# of <unk> after pre-processing Guarani: 3

texts_with_unk_normed_es = [
    text for text in tqdm(texts_with_unk_es) 
    if tokenizer.unk_token_id in tokenizer(preproc(text)).input_ids
]
print("# of <unk> after pre-processing Spanish:", len(texts_with_unk_normed_es)) # 0 - unknown after pre-processing
# of <unk> after pre-processing Spanish 1
# the same ₲ character is present in both the single spanish <unk> and in one of the three guarani ('APYRA’Ỹ' is the other two)

################################ Training the Model ################################
# optimizer
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

LANGS = [('es', 'spa_Latn'), ('gn', 'grn_Latn')]
def get_batch_pairs(batch_size, data=df_train):
    """create training batches in either direction"""
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

# set more training parameters
batch_size = 16  
max_length = 128  # token sequences will be truncated
training_steps = 40000  
losses = [] # simple avg loss tracking
MODEL_SAVE_PATH = '/mnt/storage/clouchheim/models/nllb_guarani_spanish'  # save to my models folder

#### Train Model
model.train()
x, y, loss = None, None, None
cleanup()

tq = trange(len(losses), training_steps)
for i in tq:
    xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
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
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)

###################### Evaluate and Load Model ######################
# Load trained model
model_load_name = '/mnt/storage/clouchheim/models/nllb_guarani_spanish'
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = NllbTokenizer.from_pretrained(model_load_name)

# translation function (default russian to english)
def translate(
    text, src_lang='spa_Latn', tgt_lang='grn_Latn', 
    a=32, b=3, max_input_length=1024, num_beams=8, **kwargs
):
    """Turn a text or a list of texts into a list of translations"""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True, 
        max_length=max_input_length
    )
    model.eval() # turn off training mode
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)

# Example usage:
t = 'Sus productos se mojan y no saben cómo dormirán ahí esta noche.'
print(translate(t, 'spa_Latn', 'grn_Latn')) 
# MY TRANSLATION: ['Producto-kuéra iñakÿ mbaite ha ikochô avei.']
# REAL ANSWER: ['Producto-kuéra iñakÿ mbaite ha ikochô avei.'] 

### Numerical evaluation (BLEU and ChrF++)
bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

# translate the whole df_dev
# translate into 'translated' (es -> grn)
print('starting translation from spanish to guarani')
df_dev['translated'] = df_dev['es'].apply(lambda x: translate(x, 'spa_Latn', 'grn_Latn')[0])

print(bleu_calc.corpus_score(df_dev['translated'].tolist(), [df_dev['es'].tolist()])) 
print(chrf_calc.corpus_score(df_dev['translated'].tolist(), [df_dev['es'].tolist()]))

# My results:
# BLEU = 5.43 29.9/9.7/5.1/2.6 (BP = 0.689 ratio = 0.728 hyp_len = 9176 ref_len = 12596)
# chrF2++ = 22.76 (Helsinki Standard is 34.74)

