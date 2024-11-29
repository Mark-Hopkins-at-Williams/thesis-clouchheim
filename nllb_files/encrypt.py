import argparse
import random
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

def tokenize(sents, lang, tokenizer, max_length):
    tokenizer.src_lang = lang
    tokens = tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    return tokens


def choose_split(i):
    if i > 2000:
        return 'train'
    elif i > 1000:
        return 'dev'
    else:
        return 'test'
    
    
class TokenPermuter:
    
    def __init__(self, tokenizer, corpus):    
        tokenized = tokenize(corpus, 'por_Latn', tokenizer, max_length=128)
        ids = [idx for idx in tokenized['input_ids'].unique().tolist() if 4 <= idx <= 256000]
        tokens = [(tokenizer.convert_ids_to_tokens(next_id), next_id) for next_id in ids]
        length1_tokens = [tok for tok in tokens if len(tok[0]) == 1]
        other_tokens = [tok for tok in tokens if len(tok[0]) > 1]
        underscore_tokens = [tok for tok in other_tokens if tok[0].startswith("▁")]
        no_underscore_tokens = [tok for tok in other_tokens if not tok[0].startswith("▁")]        
        self.vocab = self._permute(underscore_tokens)
        self.vocab.update(self._permute(no_underscore_tokens))
        self.vocab.update(self._permute(length1_tokens))

    def map_token_id(self, token_id):
        if token_id > 256000: # tokens above this are language ids
            return 3
        elif token_id not in self.vocab:
            return token_id
        else:
            return self.vocab[token_id]
                
    def _permute(self, indexed_tokens):        
        indices = [tok[1] for tok in indexed_tokens]
        permutation = [tok[1] for tok in indexed_tokens]
        random.shuffle(permutation)
        return dict(zip(indices, permutation))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encrypts a Finnish-English OPUS corpus")
    parser.add_argument("--num_sents", type=int, required=True, help="Number of sentences to encrypt")
    parser.add_argument("--num_langs", type=int, required=True, help="Number of artificial languages to create")
    parser.add_argument("--output_file", type=str, required=True, help="The output CSV filename")
    parser.add_argument("--parallel", type=lambda x: x.lower() == 'true', required=True, help="If artificial langs should be parallel to src lang (True/False)")
    
    args = parser.parse_args()
    base_model = "facebook/nllb-200-distilled-600M"
    src_file = '/mnt/storage/clouchheim/thesis-clouchheim/nllb_files/data/eng_por/Europarl.en-pt.en'
    tgt_file = '/mnt/storage/clouchheim/thesis-clouchheim/nllb_files/data/eng_por/Europarl.en-pt.pt'
    max_sents = args.num_sents
    num_artificial_langs = args.num_langs
    parallel = args.parallel
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tgt_sents = []
   
    # encrypt and add target language
    with open(tgt_file) as reader:
        for line in reader:
            tgt_sents.append(line.strip())
            if parallel == True:
                if len(tgt_sents) >= max_sents:
                    print('in parallel tgt')
                    break
            else:
                if len(tgt_sents) >= (max_sents * num_artificial_langs):
                    print('in not parallel tgt')
                    break

    tokenized = tokenize(tgt_sents, 'por_Latn', tokenizer, max_length=128)
    ids = [idx for idx in tokenized['input_ids'].unique().tolist() if 4 <= idx <= 256000]
    data = {'language': [], 'script': [], 'sent_id': [], 'text': [], 'split': []}
    for k in range(num_artificial_langs):  
         
        if parallel == True:
            sents = tgt_sents 
            permuter = TokenPermuter(tokenizer, sents)
            
        else:
            s = max_sents*k
            e = max_sents*(k+1)
            sents = tgt_sents[s:e] 
            tokenized = tokenize(sents, 'por_Latn', tokenizer, max_length=128)
            ids = [idx for idx in tokenized['input_ids'].unique().tolist() if 4 <= idx <= 256000]
              
        permuter = TokenPermuter(tokenizer, tgt_sents)        
        original_ids = tokenized['input_ids'].clone()
        original_ids.apply_(permuter.map_token_id)
        encrypted = tokenizer.batch_decode(original_ids, skip_special_tokens=True)
        for i, line in tqdm(enumerate(encrypted, start=0)):
            q = i
            if parallel == False:
                i += (max_sents*k)
            data['language'].append(f'p{k}r')
            data['script'].append('Latn')
            data['sent_id'].append(i)
            data['text'].append(line)
            data['split'].append(choose_split(i)) 
                       
            if q >= max_sents:
                 break
    
    
    # Work with the source language 
    with open(src_file) as reader:
        for i, line in tqdm(enumerate(reader, start = 0)):
            if parallel == True:
                q = i
            else:
                q = i % max_sents
                
            line = line.strip()
            data['language'].append('eng')
            data['script'].append('Latn')
            data['sent_id'].append(i)
            data['text'].append(line)            
            data['split'].append(choose_split(i))  
            
            if parallel == True:
                if i >= max_sents - 1:
                    break
            else:
                if i >= (max_sents * num_artificial_langs) - 1:
                    break                           
                
    df = pd.DataFrame(data)
    df.to_csv(args.output_file, index=False)
    print('wrote to:', args.output_file)
    