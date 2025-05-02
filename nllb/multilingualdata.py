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
    
class StreamingBitext(Dataset):
    def __init__(self, lang1_code, lang2_code, lang1_file, lang2_file):
        self.lang1_code = lang1_code
        self.lang2_code = lang2_code
        self.lang1_file = lang1_file
        self.lang2_file = lang2_file
        self.lang1_streamer = self.line_streamer(lang1_file)
        self.lang2_streamer = self.line_streamer(lang2_file)
        
        self.length = self._count_lines(lang1_file)
        
    def line_streamer(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield line.strip()
                
    def _count_lines(self, file_path):
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)            
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Fetch the next line from each streamer
        lang1_line = next(self.lang1_streamer)
        lang2_line = next(self.lang2_streamer)
        return lang1_line, lang2_line

class MixtureOfBitexts:
    def __init__(self, bitexts, batch_size, sampling_probs = None):
        self.bitexts = bitexts
        self.batch_size = batch_size
        self.batch_iters = [iter(DataLoader(bitext, batch_size=self.batch_size, shuffle=False, drop_last=True)) 
                            for bitext in self.bitexts]
        
        # get normalized sampleing probabilities
        if sampling_probs:
            self.sampling_probs = [p / sum(sampling_probs) for p in sampling_probs]
        else:
            self.sampling_probs = [1 / len(self.bitexts)] * len(self.bitexts)
        
    def get_language_codes(self):
        result = set()
        for bitext in self.bitexts:
            result.add(bitext.lang1_code)
            result.add(bitext.lang2_code)
        return result
        
    def next_batch(self):
        bitext_index = random.choices(range(len(self.bitexts)), weights=self.sampling_probs, k=1)[0]
        lang1_code = self.bitexts[bitext_index].lang1_code
        lang2_code = self.bitexts[bitext_index].lang2_code
        
        try:
            batch_iter = self.batch_iters[bitext_index]
            lang1_sents, lang2_sents = next(batch_iter)
        except StopIteration:
            self.batch_iters[bitext_index] = iter(DataLoader(self.bitexts[bitext_index], 
                                                             batch_size=self.batch_size, 
                                                             shuffle=False, drop_last=True))
            batch_iter = self.batch_iters[bitext_index]
            lang1_sents, lang2_sents = next(batch_iter)
        return lang1_sents, lang2_sents, lang1_code, lang2_code
       

class MultilingualCorpus:
    
    def __init__(self, data, streaming = False):
        
        self.streaming = streaming
        
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            if self.streaming:
                self.df = None
            else:
                self.df = pd.read_csv(data)
                       
    def create_bitext(self, lang1_code, lang2_code, split, lang1_file = None, lang2_file = None):
        
        if self.streaming:
            return StreamingBitext(lang1_code, lang2_code, lang1_file, lang2_file)    
        else:
            df = self.df[self.df['split']==split]
            lang1, script1 = lang1_code.split('_')
            lang2, script2 = lang2_code.split('_')
            
            df = df[(df['language'] == lang1)
                    | (df['language'] == lang2)]
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
        
    def create_gen(self, lang_code):
        column_name = self.df.columns[0]  
        sents = self.df[column_name].values.tolist()  
        none_sents = [['<gen>']] * len(sents)
        assert len(none_sents) == len(sents)
        return Bitext('<gen>', lang_code, none_sents, sents)
    
    def create_monolingual(self, lang_code, split = 'train', stream_file = None):
        if self.streaming:
            return StreamingMonolingual(lang_code, stream_file)
        else:
            df = self.df[self.df['split']==split]
            lang, script = lang_code.split('_')
            df = df[(df['language'] == lang)]
            
            used_ids = []
            sents = []
            for _, row in df.iterrows():
                if row['sent_id'] not in used_ids:
                    sents.append(row['text'])
                    used_ids.append(row['sent_id'])
            
            return Monolingual(lang_code, sents)
    
    def create_mixture_of_bitexts(self, lps, batch_size, split):
        bitexts = []
        for (l1, l2) in tqdm(lps):
            bitexts.append(self.create_bitext(l1, l2, split))
        return MixtureOfBitexts(bitexts, batch_size)
    

class Monolingual(Dataset):
    def __init__(self, lang_code, lang_sents):
        self.lang_code = lang_code
        self.lang_sents = lang_sents
        self.current_index = 0

    def __len__(self):
        return len(self.lang_sents)

    def __getitem__(self, idx):
        return self.lang_sents[idx]

    def get_next(self):
        if self.current_index < len(self.lang_sents):
            sentence = self.lang_sents[self.current_index]
            self.current_index += 1
            return sentence
        else:
            raise StopIteration
    
class StreamingMonolingual(Dataset):
    def __init__(self, lang_code, lang_file):
        self.lang_code = lang_code
        self.lang_file = lang_file
        self.streamer = self.line_streamer(lang_file)
        
        self.length = self._count_lines(lang_file)
        
    def line_streamer(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield line.strip()
                
    def _count_lines(self, file_path):
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)            
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Fetch the next line from the streamer
        return next(self.streamer)