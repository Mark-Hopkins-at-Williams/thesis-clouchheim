import random
import pandas as pd
from tqdm import tqdm
import sys

NLLB_SEED_CSV = '/mnt/storage/hopkins/data/nllb/seed/nllb_seed.csv'
LANGS = ['pbt_Arab', 'bho_Deva', 'nus_Latn', 'ban_Latn', 'dzo_Tibt', 'mni_Beng', 'lim_Latn', 
         'ltg_Latn', 'ace_Latn', 'crh_Latn', 'srd_Latn', 'taq_Latn', 'mri_Latn', 'ary_Arab', 
         'bam_Latn', 'knc_Arab', 'eng_Latn', 'knc_Latn', 'dik_Latn', 'prs_Arab', 'bjn_Arab', 
         'vec_Latn', 'fur_Latn', 'kas_Deva', 'kas_Arab', 'arz_Arab', 'lij_Latn', 'ace_Arab', 
         'bjn_Latn', 'scn_Latn', 'bug_Latn', 'lmo_Latn', 'szl_Latn', 'hne_Deva', 'fuv_Latn', 
         'taq_Tfng', 'shn_Mymr', 'mag_Deva']

NUM_SENTS = {'train': 6193, 'dev': 995, 'test': 1000}

class NllbSeedData:
    
    def __init__(self, split="train", langs=LANGS, csv_file=NLLB_SEED_CSV):
        df = pd.read_csv(csv_file)
        df = df[df['split']==split]  # restrict data to the specified split
        sent_ids = list(range(NUM_SENTS[split]))
        eng_df = df[df['language']=='eng']
        sorted_sent_ids = sorted(sent_ids, key=lambda i: len(eng_df[eng_df['sent_id']==i].iloc[0]['text']))
        self.sents_by_language = dict()
        sys.stderr.write('Loading NLLB data...\n')
        self.langs = langs
        for language_code in tqdm(langs):
            self.sents_by_language[language_code] = []
            language, script = language_code.split('_')
            language_df = df[(df['language']==language) & (df['script']==script)]
            for sent_id in sorted_sent_ids:
                sent = language_df[language_df['sent_id']==sent_id].iloc[0]['text']
                self.sents_by_language[language_code].append(sent)
        self.next_sent_index = 0
    
    def get_parallel_sents(self, language_code1, language_code2):
        return self.sents_by_language[language_code1], self.sents_by_language[language_code2]
        
    def next_batch(self, batch_size):
        lang1, lang2 = self.sample_language_pair()
        lang1_sents, lang2_sents = [], []
        while len(lang1_sents) < batch_size:
            if self.next_sent_index + 1 < len(self.sents_by_language[lang1]):
                self.next_sent_index += 1
            else:
                self.next_sent_index = 0
            lang1_sents.append(self.sents_by_language[lang1][self.next_sent_index])
            lang2_sents.append(self.sents_by_language[lang2][self.next_sent_index])
        return lang1_sents, lang2_sents, lang1, lang2
        
    def sample_language_pair(self):
        """Randomly samples a pair of languages from the NLLB-seed corpus."""
        return random.sample(self.langs, 2)
