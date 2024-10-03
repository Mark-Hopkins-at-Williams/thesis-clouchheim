from os.path import join
import pandas as pd

def load_data():

    AMERICAS_NLP_DIR = '/mnt/storage/clouchheim/data/americasnlp2024/ST1_MachineTranslation/data/'
    SRC_LANG = ('spanish', 'es')
    TGT_LANG = ('guarani', 'gn')

    # load in training data and create df_train
    with open(join(AMERICAS_NLP_DIR, f'{TGT_LANG[0]}-{SRC_LANG[0]}', f'train.{SRC_LANG[1]}'), 'r', encoding='utf-8') as f:
        src_sentences = f.readlines()
    with open(join(AMERICAS_NLP_DIR, f'{TGT_LANG[0]}-{SRC_LANG[0]}', f'train.{TGT_LANG[1]}'), 'r', encoding='utf-8') as f:
        tgt_sentences = f.readlines()
        
    src_sentences = [line.strip() for line in src_sentences]
    tgt_sentences = [line.strip() for line in tgt_sentences]
        
    # column names: ['id', 'es', 'gn']
    # shape: (26302, 3)
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
        
    # column names: ['id', 'es', 'gn']
    # shape: (995, 3)
    df_dev = pd.DataFrame({    
        'id': range(len(src_sentences_dev)),
        'src': src_sentences_dev,
        'tgt': tgt_sentences_dev
    })
    return df_train, df_dev

def guarani_sents():
    df_train, df_dev = load_data()
    for line in df_train['src']:
        yield(line)
        
def chinese_sents():
    with open('baidu.txt') as reader:
        for line in reader:
            line = line.strip()
            yield line
        
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

char_counts = dict()
normalizer = 0
for line in chinese_sents():
    for char in line:
        normalizer += 1
        if char not in char_counts:
            char_counts[char] = 0
        char_counts[char] += 1
char_probs = {char: char_counts[char] / normalizer for char in char_counts}
    
print(char_probs)