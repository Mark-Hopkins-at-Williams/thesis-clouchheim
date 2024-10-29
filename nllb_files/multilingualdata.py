import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Bitext(Dataset):
    def __init__(self, lang1_code, lang2_code, lang1_sents, lang2_sents):
        self.lang1_code = lang1_code
        self.lang2_code = lang2_code
        self.lang1_sents = lang1_sents
        self.lang2_sents = lang2_sents
        
    def __len__(self):
        return len(self.lang1_sents)

    def __getitem__(self, idx):
        return self.lang1_sents[idx], self.lang2_sents[idx]
    

class MixtureOfBitexts:
    def __init__(self, bitexts, batch_size):
        self.bitexts = bitexts
        self.batch_size = batch_size
        self.batch_iters = [iter(DataLoader(bitext, batch_size=self.batch_size, shuffle=True, drop_last=True)) 
                            for bitext in self.bitexts]
        
    def get_language_codes(self):
        result = set()
        for bitext in self.bitexts:
            result.add(bitext.lang1_code)
            result.add(bitext.lang2_code)
        return result
        
    def next_batch(self):
        bitext_index = random.randint(0, len(self.bitexts)-1)
        lang1_code = self.bitexts[bitext_index].lang1_code
        lang2_code = self.bitexts[bitext_index].lang2_code
        try:
            batch_iter = self.batch_iters[bitext_index]
            lang1_sents, lang2_sents = next(batch_iter)
        except StopIteration:
            self.batch_iters[bitext_index] = iter(DataLoader(self.bitexts[bitext_index], 
                                                             batch_size=self.batch_size, 
                                                             shuffle=True, drop_last=True))
            batch_iter = self.batch_iters[bitext_index]
            lang1_sents, lang2_sents = next(batch_iter)
        return lang1_sents, lang2_sents, lang1_code, lang2_code
       

class MultilingualCorpus:
    
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
                
    def create_bitext(self, lang1_code, lang2_code, split):
        df = self.df[self.df['split']==split]
        print(self.df.size)
        lang1, script1 = lang1_code.split('_')
        lang2, script2 = lang2_code.split('_')
        
        df = df[((df['language'] == lang1) & (df['script'] == script1)) 
                | ((df['language'] == lang2) & (df['script'] == script2))]
        
        sents = dict()
        for _, row in df.iterrows():
            if row['sent_id'] not in sents:
               sents[row['sent_id']] = []
            lang_code = f"{row['language']}_{row['script']}"            
            sents[row['sent_id']].append((lang_code, row['text']))
        sents = {key: sents[key] for key in sents if len(sents[key]) > 1}
        lang1_sents, lang2_sents = [], []
        for key in sents:
            lang1_sent = None
            lang2_sent = None
            for (lang_code, sent) in sents[key]:
                if lang_code == lang1_code:
                    lang1_sent = sent
                elif lang_code == lang2_code:
                    lang2_sent = sent
            if lang1_sent is not None and lang2_sent is not None:
                lang1_sents.append(lang1_sent)
                lang2_sents.append(lang2_sent)
        return Bitext(lang1_code, lang2_code, lang1_sents, lang2_sents)
    
    def create_mixture_of_bitexts(self, lps, batch_size):
        bitexts = []
        for (l1, l2) in tqdm(lps):
            bitexts.append(self.create_bitext(l1, l2, 'train'))
        return MixtureOfBitexts(bitexts, batch_size)
    
    
class WeightedMixtureOfBitexts:
    
    def __init__(self, bitexts, weights, batch_size):
        self.bitexts = bitexts
        self.weights = np.array(weights) / sum(weights)  # Normalize weights
        self.batch_size = batch_size
        self.batch_iters = [iter(DataLoader(bitext, batch_size=self.batch_size, shuffle=True, drop_last=True)) 
                           for bitext in self.bitexts]
    
    def get_language_codes(self):
        result = set()
        for bitext in self.bitexts:
            result.add(bitext.lang1_code)
            result.add(bitext.lang2_code)
        return result
    
    def next_batch(self):
        bitext_index = np.random.choice(len(self.bitexts), p=self.weights)
        lang1_code = self.bitexts[bitext_index].lang1_code
        lang2_code = self.bitexts[bitext_index].lang2_code
        
        try:
            batch_iter = self.batch_iters[bitext_index]
            lang1_sents, lang2_sents = next(batch_iter)
        except StopIteration:
            self.batch_iters[bitext_index] = iter(DataLoader(self.bitexts[bitext_index], 
                                                           batch_size=self.batch_size, 
                                                           shuffle=True, drop_last=True))
            batch_iter = self.batch_iters[bitext_index]
            lang1_sents, lang2_sents = next(batch_iter)
            
        return lang1_sents, lang2_sents, lang1_code, lang2_code

class ThreePhaseDataCreator:
    def __init__(self, opus_csv, americas_csv):
        self.opus_corpus = MultilingualCorpus(opus_csv)
        self.americas_corpus = MultilingualCorpus(americas_csv)
        self.opus_train = self.opus_corpus.create_bitext('spa_Latn', 'eng_Latn', 'train')
        self.opus_dev = self.opus_corpus.create_bitext('spa_Latn', 'eng_Latn', 'dev')
        
    def create_phase1_mixture(self, batch_size):
        """
        Phase 1: 91% Spanish-English, 9% indigenous languages
        """
        bitexts = []
        weights = []
        
        es_en = self.opus_train
        bitexts.append(es_en)
        weights.append(0.91) # 91% total for Spanish-English
        
        # Add indigenous languages data (9% split equally)
        indigenous_pairs = [
            ('aym_Latn', 'spa_Latn'), ('bzd_Latn', 'spa_Latn'),
            ('cni_Latn', 'spa_Latn'), ('grn_Latn', 'spa_Latn'),
            ('hch_Latn', 'spa_Latn'), ('nah_Latn', 'spa_Latn'),
            ('oto_Latn', 'spa_Latn'), ('quy_Latn', 'spa_Latn'),
            ('tar_Latn', 'spa_Latn'), ('shp_Latn', 'spa_Latn'),
            ('ctp_Latn', 'spa_Latn')
        ]
        
        indigenous_weight = 0.09 / len(indigenous_pairs)
        for pair in indigenous_pairs:
            bitext = self.americas_corpus.create_bitext(pair[0], pair[1], 'train')
            bitexts.append(bitext)
            weights.append(indigenous_weight)
            
        return WeightedMixtureOfBitexts(bitexts, weights, batch_size)
    
    def create_phase2_mixture(self, batch_size):
        """
        Phase 2: 37% Spanish-English, 63% indigenous languages
        """
        bitexts = []
        weights = []
        
        es_en = self.opus_train
        bitexts.append(es_en)
        weights.append(0.37) # 37% total for Spanish-English
        
        # Add indigenous languages data (63% split equally)
        indigenous_pairs = [
            ('aym_Latn', 'spa_Latn'), ('bzd_Latn', 'spa_Latn'),
            ('cni_Latn', 'spa_Latn'), ('grn_Latn', 'spa_Latn'),
            ('hch_Latn', 'spa_Latn'), ('nah_Latn', 'spa_Latn'),
            ('oto_Latn', 'spa_Latn'), ('quy_Latn', 'spa_Latn'),
            ('tar_Latn', 'spa_Latn'), ('shp_Latn', 'spa_Latn'),
            ('ctp_Latn', 'spa_Latn')
        ]
        
        indigenous_weight = 0.63 / len(indigenous_pairs)
        for pair in indigenous_pairs:
            bitext = self.americas_corpus.create_bitext(pair[0], pair[1], 'train')
            bitexts.append(bitext)
            weights.append(indigenous_weight)
            
        return WeightedMixtureOfBitexts(bitexts, weights, batch_size)
    
    def create_phase3_data(self, target_lang, batch_size):
        """
        Phase 3: Language-specific fine-tuning (40% English, 60% target language)
        """
        bitexts = []
        weights = []
        
        # Add Spanish-English data (40%)
        es_en = self.opus_train
        bitexts.append(es_en)
        weights.append(0.4)
        
        # Add target language data (60%)
        target_bitext = self.americas_corpus.create_bitext(f'{target_lang}_Latn', 'spa_Latn', 'train')
        bitexts.append(target_bitext)
        weights.append(0.6)
        
        return WeightedMixtureOfBitexts(bitexts, weights, batch_size) 
    
    def create_concatenated_dev_set(self):
        """
        Create a concatenated development set with 50% Spanish-English and 50% indigenous languages.
        """
        dev_texts = []
        
        # Combine Spanish-English data (you can adjust proportions here if desired)
        dev_texts.extend(self.opus_dev)
        
        # Indigenous languages development data (split equally among 11 languages)
        indigenous_pairs = [
            ('aym_Latn', 'spa_Latn'), ('bzd_Latn', 'spa_Latn'),
            ('cni_Latn', 'spa_Latn'), ('grn_Latn', 'spa_Latn'),
            ('hch_Latn', 'spa_Latn'), ('nah_Latn', 'spa_Latn'),
            ('oto_Latn', 'spa_Latn'), ('quy_Latn', 'spa_Latn'),
            ('tar_Latn', 'spa_Latn'), ('shp_Latn', 'spa_Latn'),
            ('ctp_Latn', 'spa_Latn')
        ]
        
        # Add indigenous language data with equal portions for each language pair
        for pair in indigenous_pairs:
            dev_text = self.americas_corpus.create_bitext(pair[0], pair[1], 'dev')
            dev_texts.extend(dev_text)
        
        # Return the concatenated list
        return dev_texts
    