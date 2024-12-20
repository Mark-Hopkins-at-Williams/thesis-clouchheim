import pandas as pd
import unicodedata
import os
import re

# This program makes csv files containing all the training and dev data for each of the language
# pairs in the americasnlp dataset. each csv has the headers
#               [src_lang_code], spa, split

# where split is either "train" or "dev"

AMERICASNLP_CODES = { # these are just the codes that americasnlp uses for their files
    "ashaninka": "cni",
    "bribri": "bzd",
    "guarani": "gn",
    "quechua": "quy",
    "aymara": "aym",
    "shipibo_konibo": "shp",
    "chatino": "ctp",
    "hñähñu": "oto",
    "nahuatl": "nah",
    "raramuri": "tar",
    "wixarika": "hch"
    }

parent_path = "/mnt/storage/hopkins/data/americasnlp2024/ST1_MachineTranslation/data"

sent_ids = dict()


def scrape(path, code, split, info):
    with open(path + f'/{split}.es', 'r') as file:
        es_sents = file.readlines()
    es_sents = [line.strip() for line in es_sents]
    with open(path + f'/{split}.' + code, 'r') as file:
        src_sents = file.readlines()
    src_sents = [line.strip() for line in src_sents]
    for es, src in zip(es_sents, src_sents):
        if len(src) > 0 and len(es) > 0:
            if es not in sent_ids:
                sent_ids[es] = len(sent_ids)
                info.languages.append('spa')
                info.scripts.append('Latn')
                info.sent_ids.append(sent_ids[es])
                info.texts.append(es)
                info.splits.append(split)
            info.languages.append(code)
            info.scripts.append('Latn')
            info.sent_ids.append(sent_ids[es])
            info.texts.append(src)
            info.splits.append(split)


class DataFrameInfo:
    def __init__(self):
        self.languages = []
        self.scripts = []
        self.sent_ids = []
        self.texts = []
        self.splits = []


if __name__ == "__main__":
    info = DataFrameInfo()
    for subdir, dirs, files in os.walk(parent_path):
        if subdir == parent_path:
            for dir_name in dirs:
                print(dir_name)
                dir_path = os.path.join(subdir, dir_name)
                language = dir_path.split("/")[-1].split("-")[0] #identify the american language

                scrape(dir_path, AMERICASNLP_CODES[language], 'train', info)
                scrape(dir_path, AMERICASNLP_CODES[language], 'dev', info)
    df = pd.DataFrame({'language': info.languages,
                       'script': info.scripts,
                       'sent_id': info.sent_ids,
                       'text': info.texts,
                       'split': info.splits})
    
    # account for difference in nlp suffix and guarani nllb tag (grn_Latn) and aymaere nllb tag (ayr_Latn)
    
    # for ctp, cahnge all super script to normal letters
    
    # for nah, oto detokenize to put in line with rest of vocab
    
    # change all punctiation not in NLLB-200s vocab
    
    df['language'] = df['language'].replace('gn', 'grn')
    df['language'] = df['language'].replace('ayr', 'aym')
    
    # normalize super script and punctuation and special characters
    char_map = {
    'ᴬ': 'A', 'ᴮ': 'B', 'ᶜ': 'C', 'ᴰ': 'D', 'ᴱ': 'E', 'ᴴ': 'H', 'ᴵ': 'I', 'ᴶ': 'J', 
    'ᴷ': 'K', 'ᴸ': 'L', 'ᴹ': 'M', 'ᴺ': 'N', 'ᴼ': 'O', 'ᴾ': 'P', 'ʳ': 'R', 
    'ᵀ': 'T', 'ᵁ': 'U', 'ⱽ': 'V', 'ʷ': 'W', 'ⁱ': 'i', 'ⁿ': 'n', '⁺': '+', 
    '⁻': '-', '⁼': '=', '⁽': '(', '⁾': ')', 'ᶠ': 'F', 'ᴳ': 'G', 'ᴿ': 'R',
    '“':'"', '”':'"', '—': '', 'č':'c', 'ç': 'c', '""': '"', '’':"'", '‘':"'",
    '–': '', '«': '"', 'ⴘ':'', 'ô':'o', 'ƌ':'d', 'ȥ':'z', '»':'"', 'Ƌ':'d',
    'ᶎ':'z', 'o̮':'o', 'î':'i', 'u̡':'u', '„':'"', '‟':'"', 'ḱ':'k', '₲':'$'
    }
    
    # this works for one line
    def replace_characters(text):
        # Normalize the text to a consistent form (NFD separates diacritics from base characters)
        text = unicodedata.normalize('NFD', text)
        # Replace characters based on the map
        for char, replacement in char_map.items():
            text = text.replace(char, replacement)
        # Re-normalize back to NFC to combine characters if needed
        return unicodedata.normalize('NFC', text)
    
    def clean_text(text):
        # Remove spaces inside parentheses
        text = re.sub(r'\s*\(\s*', '(', text)  # Remove spaces before '('
        text = re.sub(r'\s*\)\s*', ')', text)  # Remove spaces after ')'
    
        # Remove spaces before punctuation (e.g., ':', ',', '.')
        text = re.sub(r'\s+([:,.\'])', r'\1', text)  # No space before punctuation
        return text

    df['text'] = df['text'].str.strip()
    df['text'] = df['text'].apply(clean_text)
    
    # only apply normalization to 
    languages_to_normalize = ['oto', 'ctp', 'nah', 'shp']
    df.loc[df['language'].isin(languages_to_normalize), 'text'] = df.loc[df['language'].isin(languages_to_normalize), 'text'].apply(replace_characters)
    
    
    df.to_csv("americas_nlp_data.csv", index = False)