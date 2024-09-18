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
from datasets import load_dataset
from collections import Counter

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

#test - russain to english
inputs = tokenizer(text="поля озарились утренним солнцем", return_tensors="pt")
translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
print(tokenizer.decode(translated_tokens[0], skip_special_tokens=True)) # Return: The fields were lit by the morning sun

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
tyv_wiki = load_dataset("graelo/wikipedia", "20230601.tyv")
tyv_wiki
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'url', 'title', 'text'],
#         num_rows: 3459
#     })
# })
print(sum(len(t) for t in tyv_wiki['train']['text']))  # 7568832
print(sum(len(t) for t in trans_df.tyv.dropna()))      # 3573803

all_texts = tyv_wiki['train']['text'] + df_train.tyv.dropna().tolist()
all_text_normalized = [preproc(t) for t in tqdm(all_texts)]
chars_cnt = Counter(c for t in all_text_normalized for c in t)
required_chars = ''.join([
    k for k, v in chars_cnt.most_common() 
    if v >= 3 and k not in ' '
])

import sentencepiece as spm
all_texts_file = 'myv_texts_plain.txt'
SPM_PREFIX = 'spm_tyvan_16k'
with open(all_texts_file, 'w') as f:
    for i, text in enumerate(all_texts):
        print(text, file=f)

spm.SentencePieceTrainer.train(
    input=all_texts_file,
    model_prefix=SPM_PREFIX,
    vocab_size=2**14,  # 16K
    character_coverage = 1,
    num_threads=16,
    train_extremely_large_corpus=False,
    add_dummy_prefix=False,
    max_sentencepiece_length=128,
    max_sentence_length=4192*4,
    pad_id=0,
    eos_id=1,
    unk_id=2,
    bos_id=-1,
    required_chars=required_chars,
)

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
# At this step, the code may throw an error about protobuf. Do as it tells.
from transformers import NllbTokenizer

# reading the NLLB and the Tyvan sentencepiece models into a native format
tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
sp_trained = spm.SentencePieceProcessor(model_file=f'{SPM_PREFIX}.model')
added_spm = sp_pb2_model.ModelProto()
added_spm.ParseFromString(sp_trained.serialized_model_proto())
old_spm = sp_pb2_model.ModelProto()
old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

# adding the missing tokens to the NLLB sentencepiece model
nllb_tokens_set = {p.piece for p in old_spm.pieces}
prev_min_score = old_spm.pieces[-1].score
for p in added_spm.pieces:
    piece = p.piece
    # !!! THIS FIX WAS ADDED LATER; it is required for CT2 compatibility !!!
    # 1 is ordinary token, non-1 is special token; we don't want to copy the special tokens
    if p.type != 1:
        continue
    if piece not in nllb_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        # for all new tokens, I'll set a lower score (priority)
        new_p.score = p.score + prev_min_score
        old_spm.pieces.append(new_p)

# saving the result to disk
NEW_SPM_NAME = 'spm_nllb_tyvan_268k.model'
with open(NEW_SPM_NAME, 'wb') as f:
    f.write(old_spm.SerializeToString())
    
from transformers import AutoModelForSeq2SeqLM
model_name = 'facebook/nllb-200-distilled-600M'

# loading the tokenizers
tokenizer_old = NllbTokenizer.from_pretrained(model_name)
tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=NEW_SPM_NAME)
print(len(tokenizer_old), len(tokenizer)) # 256204, 268559
added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
print(len(added_vocab))  # 12355

# loading and resizing the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# re-initializing the new embeddings
for t in tqdm(added_vocab):
    tt = tokenizer_old(t, add_special_tokens=False).input_ids
    if len(tt) == 0:
        tt = [tokenizer_old.unk_token_id]
    idx = tokenizer.convert_tokens_to_ids(t)
    model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)

################################ Adding new Langauge Tag ################################
def fix_tokenizer(tokenizer, new_lang='tyv_Cyrl'):
    """
    Add a new language token to the tokenizer vocabulary 
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {} 


model_name = "facebook/nllb-200-distilled-600M"
# loading the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# patching them
fix_tokenizer(tokenizer)
model.resize_token_embeddings(len(tokenizer))

# fixing the new/moved token embeddings in the model
added_token_id = tokenizer.convert_tokens_to_ids('tyv_Cyrl')
similar_lang_id = tokenizer.convert_tokens_to_ids('kir_Cyrl')
embeds = model.model.shared.weight.data
# moving the embedding for "mask" to its new position
embeds[added_token_id+1] =embeds[added_token_id]
# initializing new language token with a token of a similar language
embeds[added_token_id] = embeds[similar_lang_id]

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

###################### Evaluate and Load Model ######################

# Load trained model
model_load_name = '/mnt/storage/clouchheim/models/nllb_tyvan_russian_v1'
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = NllbTokenizer.from_pretrained(model_load_name)
#fix_tokenizer(tokenizer)

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