import sacrebleu
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_load_name = 'facebook/nllb-200-distilled-600M'
#model_load_name = '/mnt/storage/hopkins/models/nllb-rus-tyv-v1'
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
model_tok_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_tok_name)
#tsv_file = '/mnt/storage/hopkins/thesis/data/rus_tyv_parallel_50k.tsv'
LANGS = [('es', 'spa_Latn'), ('gn', 'grn_Latn')]

def translate(
    text, src_lang, tgt_lang, 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    """Translates a string or a list of strings."""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    model.eval() # turn off training mode
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True, 
        max_length=max_input_length
    )
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)
    

def batched_translate(texts, batch_size=16, **kwargs):
    """Translate texts in batches of similar length"""
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]


bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

#trans_df = pd.read_csv(tsv_file, sep="\t")

#df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
#df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
#df_test = trans_df[trans_df.split=='test'].copy()   # 500 items

from americasnlpdata import load_data

df_train, df_dev = load_data()


rus_translations = batched_translate(df_dev['src'].tolist(), src_lang='spa_Latn', tgt_lang='grn_Latn')

print(bleu_calc.corpus_score(rus_translations, [df_dev['tgt'].tolist()]))
print(chrf_calc.corpus_score(rus_translations, [df_dev['tgt'].tolist()]))