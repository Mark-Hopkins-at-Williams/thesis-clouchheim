import evaluate
import os
import sys
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from multilingualdata import MultilingualCorpus

def translate(
    text, tokenizer, model, 
    src_lang, tgt_lang, 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    model.eval()
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True, 
        max_length=max_input_length
    )
    generation_config = GenerationConfig(
        max_length=200
    )
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        generation_config=generation_config,
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result[:, 2:], skip_special_tokens=True) # TODO: the slicing shouldn't be necessary if special tokens are properly recognized


def batched_translate(texts, batch_size=16, **kwargs):
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i: i+batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]


def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result = bleu_calc.compute(predictions = candidate_translations, references = reference_translations)
    chrf_result = chrf_calc.compute(predictions = candidate_translations, references = reference_translations)
    return round(bleu_result["score"], 3), round(chrf_result["score"], 3)
    
    
if __name__ == "__main__":
    base_model = "./f2n_model-v0" 
    corpus = MultilingualCorpus('eng_to_f2n.csv')
    lps = [('eng_Latn', 'f0n_Latn'), ('eng_Latn', 'f1n_Latn')]
    bleu_scores = []
    chrf_scores = []
    for lp in lps:
        bitext = corpus.create_bitext(lp[0], lp[1], 'test')
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        model.cuda()
        src_texts = bitext.lang1_sents
        tgt_texts = bitext.lang2_sents
        candidate_translations = batched_translate(src_texts, tokenizer=tokenizer, model=model, src_lang=lp[0], tgt_lang=lp[1])        
        bleu, chrf = evaluate_translations(candidate_translations, tgt_texts)
        with open(os.path.join(base_model, f'candidates.{lp[1]}'), 'w') as writer:
            for candidate in candidate_translations:
                writer.write(f'{candidate}\n')
        with open(os.path.join(base_model, f'references.{lp[1]}'), 'w') as writer:
            for ref in tgt_texts:
                writer.write(f'{ref}\n')
        bleu_scores.append(bleu)
        chrf_scores.append(chrf)
    with open(os.path.join(base_model, 'scores.csv'), 'w') as writer:
        writer.write(','.join(['src', 'tgt', 'bleu', 'chrf']) + '\n')
        for i, (src, tgt) in enumerate(lps):
            writer.write(','.join([src, tgt, str(bleu_scores[i]), str(chrf_scores[i])]) + '\n')