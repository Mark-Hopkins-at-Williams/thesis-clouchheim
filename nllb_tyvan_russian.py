########### Practice with nllb fine tuning following medium blog ###########

# Import libraries
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

################################ Load data ################################

# column names: ['row_id', 'ind', 'tyv', 'ru', 'split']
# shape: (50000, 5)
trans_df = pd.read_csv('/mnt/storage/clouchheim/data/rus_tyv_parallel_50k.tsv', sep="\t")

# split into train, dev, test
df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
df_test = trans_df[trans_df.split=='test'].copy()   # 500 items

#### Load model and Tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer.src_lang = "rus_Cyrl"

# test - russain to english
#inputs = tokenizer(text="поля озарились утренним солнцем", return_tensors="pt")
#translated_tokens = model.generate(
    #**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
#print(tokenizer.decode(translated_tokens[0], skip_special_tokens=True)) # Return: The fields were lit by the morning sun

################################ Determine if this tokenizer is adequet for Tyvan ################################
def word_tokenize(text):
    """
    Split a text into words, numbers, and punctuation marks
    (for languages where words are separated by spaces)
    """
    return re.findall('(\w+|[^\w\s])', text)

# tokenize and work_tokenize
smpl = df_train.sample(10000, random_state=1)
smpl['rus_toks'] = smpl.ru.apply(tokenizer.tokenize)
smpl['tyv_toks'] = smpl.tyv.apply(tokenizer.tokenize)
smpl['rus_words'] = smpl.ru.apply(word_tokenize)
smpl['tyv_words'] = smpl.tyv.apply(word_tokenize)

# see number of tokens per word for russian and tyvan
stats = smpl[
    ['rus_toks', 'tyv_toks', 'rus_words', 'tyv_words']
].applymap(len).describe()
print("# of tokens per word")
print("Russian:", stats.rus_toks['mean'] / stats.rus_words['mean'])  # number of tokens per word Russian: 2.0349
print("Tyvan:", stats.tyv_toks['mean'] / stats.tyv_words['mean'])  # number of tokens per word Tyvan: 2.4234

# see number of unknowns 
texts_with_unk = [
    text for text in tqdm(trans_df.tyv) 
    if tokenizer.unk_token_id in tokenizer(text).input_ids
]
print("# of <unk> before pre-processing:", len(texts_with_unk)) # 163 - number of tyvan tokens that are unknown (mostly from unknown characters)
s = random.sample(texts_with_unk, 5)
print(s) 

#### Preprocess Data 

mpn = MosesPunctNormalizer(lang="en")
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
texts_with_unk_normed = [
    text for text in tqdm(texts_with_unk) 
    if tokenizer.unk_token_id in tokenizer(preproc(text)).input_ids
]
print("# of <unk> after pre-processing", len(texts_with_unk_normed)) # 0 - unknown after pre-processing


################################ Expanding Vocabulary ################################
#TODO(optional): complete this section 

################################ Adding new Langauge Tag ################################
#TODO(optional): complete this section

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

LANGS = [('ru', 'rus_Cyrl'), ('tyv', 'tyv_Cyrl')]
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
training_steps = 60000  
losses = [] # simple avg loss tracking
MODEL_SAVE_PATH = '/mnt/storage/clouchheim/models/nllb_tyvan_russian'  # save to my models folder

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
        
################################ Evaluating the Model ################################

# Load trained model
model_load_name = '/mnt/storage/clouchheim/models/nllb_tyvan_russian'
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = NllbTokenizer.from_pretrained(model_load_name)
fix_tokenizer(tokenizer)

# translation function (default russian to english)
def translate(
    text, src_lang='rus_Cyrl', tgt_lang='eng_Latn', 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
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
t = 'мөңгүн үр чыткаш карарар'
print(translate(t, 'tyv_Cyrl', 'rus_Cyrl')) # test tyvan to russain for t
# ['серебро от времени чернеет']

### Numerical evaluation (BLEU and ChrF++)
bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

print(bleu_calc.corpus_score(df_dev['rus_translated'].tolist(), [df_dev['ru'].tolist()]))
print(chrf_calc.corpus_score(df_dev['rus_translated'].tolist(), [df_dev['ru'].tolist()]))
print(bleu_calc.corpus_score(df_dev['tyv_translated'].tolist(), [df_dev['tyv'].tolist()]))
print(chrf_calc.corpus_score(df_dev['tyv_translated'].tolist(), [df_dev['tyv'].tolist()]))

################################ Publishing ################################
#TODO: complete this section

