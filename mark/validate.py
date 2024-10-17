import sacrebleu
from nllbseed import NllbSeedData
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def translate(
    text, tokenizer, model, 
    src_lang, tgt_lang, 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    model.eval() # turn off training mode
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
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
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]


def evaluate(candidate_translations, reference_translations):
    bleu_calc = sacrebleu.BLEU()
    chrf_calc = sacrebleu.CHRF( # I think this is the official metric of AmericasNLP
        word_order=0, 
        char_order=6, 
        lowercase=False, 
        whitespace=False
    )
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result  = str(bleu_calc.corpus_score(candidate_translations, reference_translations))
    chrf_result = str(chrf_calc.corpus_score(candidate_translations, reference_translations))
    print(bleu_result)
    print(chrf_result)
    
    
if __name__ == "__main__":
    base_model = "facebook/nllb-200-distilled-600M" 
    dev_data = NllbSeedData(split="dev", langs=['eng_Latn', 'lij_Latn'])
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    model.cuda()
    src_texts = dev_data.get_sents('lij_Latn')[:50]
    tgt_texts = dev_data.get_sents('eng_Latn')[:50]
    candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang='lij_Latn', tgt_lang='eng_Latn')
    for candidate, gold in zip(candidate_translations, tgt_texts):
        print('-'*5)
        print(f'candidate: {candidate}')
        print(f'gold:      {gold}')
        
    evaluate(candidate_translations, tgt_texts)