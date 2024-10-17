import random
import pandas as pd
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
  