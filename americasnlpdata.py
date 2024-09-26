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
