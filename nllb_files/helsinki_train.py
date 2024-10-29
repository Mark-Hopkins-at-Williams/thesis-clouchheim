import gc
import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from finetune import tokenize
from validate import evaluate_translations, log_evaluation, batched_translate
from multilingualdata import ThreePhaseDataCreator, MultilingualCorpus
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

class ThreePhaseTrainer:
    def __init__(
        self,
        model_name,
        device,
        max_length=128,
        report_every=100,
        validate_every=1000,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_config(config)
        self.device = device
        self.max_length = max_length
        self.report_every = report_every
        self.validate_every = validate_every
        
        if device == "cuda":
            self.model.cuda()
    
    def train_step(self, batch, optimizer, max_length):
        self.model.train()
        lang1_sents, lang2_sents, lang1, lang2 = batch
        
        try:
            x = tokenize(lang1_sents, lang1, self.tokenizer, 128).to(self.device)
            y = tokenize(lang2_sents, lang2, self.tokenizer, 128, alt_pad_token=-100).to(self.device)
            loss = self.model(**x, labels=y.input_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return loss.item()
        except RuntimeError:
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            print('GPU out of memory! Performing garbage collection.')
            return None

    def train_phase(
        self,
        data_iterator,
        dev_data,
        steps,
        save_dir,
        phase_name,
        optimizer=None,
        scheduler=None,
    ):
        if optimizer is None:
            optimizer = Adafactor(
                [p for p in self.model.parameters() if p.requires_grad],
                scale_parameter=False,
                relative_step=False,
                lr=1e-4,
                clip_threshold=1.0,
                weight_decay=1e-3,
            )
        if scheduler is None:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

        train_losses = []
        best_chrf = 0
        best_step = 0
        patience = 20000
        max_length = 128
        best_model_path = None

        for step in tqdm(range(steps)):
            batch = data_iterator.next_batch()
            loss = self.train_step(batch, optimizer, max_length)
            
            if loss is not None:
                train_losses.append(loss)
                scheduler.step()

            if step % self.report_every == 0 and step > 0:
                avg_loss = np.mean(train_losses[-self.report_every:])
                print(f'{phase_name} step {step} (train): {avg_loss}')
                sys.stdout.flush()

            if step % self.validate_every == 0:
                print(f"Validating {phase_name} on dev set...")
                src_texts, tgt_texts = dev_data.lang1_sents, dev_data.lang2_sents
                candidate_translations = batched_translate(src_texts, tokenizer=self.tokenizer, model=self.model, src_lang=dev_data.lang1_code, tgt_lang=dev_data.lang2_code)
                for candidate, gold in zip(candidate_translations[:5], tgt_texts[:5]):
                    print('-'*5)
                    print(f'candidate: {candidate}')
                    print(f'gold:      {gold}')
                bleu, chrf = evaluate_translations(candidate_translations, tgt_texts)
                print(f"Step {step} - BLEU: {bleu}, chrF: {chrf}")

                if chrf > best_chrf:
                    best_chrf = chrf
                    best_step = step
                    if phase_name == "phase2":
                        save_path = os.path.join(save_dir, f"{phase_name}_step_{step}")
                    else:
                        save_path = os.path.join(save_dir, f"{phase_name}")
                    os.makedirs(save_path, exist_ok=True)
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    best_model_path = save_path
                    print(f"Saved new best model for {phase_name} to {save_path}")

            if step - best_step >= patience:
                print(f'No improvement in {patience} steps. Stopping {phase_name} training.')
                break

        return best_model_path, best_chrf

    def train_all_phases(
        self,
        phase1_data,  # 91% es-en, 9% indigenous
        phase2_data,  # 37% es-en, 63% indigenous
        phase3_data_generator,  # Function that returns data iterator for specific language
        phase1_dev_data, 
        phase2_dev_data,
        phase3_dev_corpus,
        save_dir,
        target_languages,
    ):
        # Phase 1: Initial training
        print("Starting Phase 1 training...")
        phase1_path, phase1_chrf = self.train_phase(
            phase1_data,
            phase1_dev_data,
            steps=100000,
            save_dir=save_dir,
            phase_name="phase1"
        )

        # Phase 2: Balanced training
        print("Starting Phase 2 training...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(phase1_path)
        self.model.to(self.device)
        phase2_path, phase2_chrf = self.train_phase(
            phase2_data,
            phase2_dev_data,
            steps=200000,
            save_dir=save_dir,
            phase_name="phase2"
        )

        # Phase 3: Language-specific fine-tuning
        results = {}
        for lang in target_languages:
            print(f"Starting Phase 3 training for {lang}...")
            # Load best model from Phase 2
            self.model = AutoModelForSeq2SeqLM.from_pretrained(phase2_path)
            self.model.to(self.device)
            
            # Get language-specific training data (40% English, 60% target language)
            print(f"done making phase 3 lang specific data data (40% eng, 60% {lang})")
            lang_specific_data = phase3_data_generator.create_phase3_data(lang, 2)
            dev_data = phase3_dev_corpus.create_bitext('spa_Latn', f'{lang}_Latn', 'dev')
            
            phase3_path, phase3_chrf = self.train_phase(
                lang_specific_data,
                dev_data,
                steps=12000,
                save_dir=os.path.join(save_dir, f"phase3_{lang}"),
                phase_name=f"phase3_{lang}"
            )
            
            # Compare with Phase 2 results
            if phase3_chrf > phase2_chrf:
                results[lang] = {"path": phase3_path, "chrf": phase3_chrf}
            else:
                results[lang] = {"path": phase2_path, "chrf": phase2_chrf}
                
        return results


if __name__ == "__main__":
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(description="Three-phase training for multilingual translation")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training")
    args = parser.parse_args()
    
    print(f"Starting Training Script for {args.save_dir}")
    print("Creating phased trainer object")
    sys.stdout.flush()
    trainer = ThreePhaseTrainer(
        model_name="facebook/nllb-200-distilled-600M",
        device=args.device
    )
    
    # load in phase data
    print("Creating data in specific ratios")
    sys.stdout.flush()
    
    americas_csv = "americas_nlp_data.csv"
    opus_csv = "opus_test.csv"
    '''
    creator = ThreePhaseDataCreator("opus_data.csv", "americas_nlp_data.csv")
    phase1_data = creator.create_phase1_mixture(batch_size=2)
    print("Done making phase 1 data (91% eng, 9%anlp)")
    sys.stdout.flush()
    
    phase2_data = creator.create_phase2_mixture(batch_size=2)
    print("Done making phase 2 data (37% eng, 63%anlp)")
    sys.stdout.flush()
    
    phase3_data = creator
    
    # load in dev data
    print("creating dev data")
    sys.stdout.flush()
    americasnlp_corpus = MultilingualCorpus(americas_csv)
    '''
    opus_corpus = MultilingualCorpus(opus_csv)
    phase1_dev_data = opus_corpus.create_bitext("spa_Latn", "eng_Latn", 'dev')
    print(len(phase1_dev_data))
    
    phase2_dev_data = creator.create_concatenated_dev_set() # 50/50 eng-spa and indegenous
    creator = ThreePhaseDataCreator("opus_data.csv", "americas_nlp_data.csv") ##########
    
    phase3_dev_corpus = americasnlp_corpus
    
    print("Ready to train")
    results = trainer.train_all_phases(
        phase1_data,
        phase2_data,
        phase3_data,
        phase1_dev_data,
        phase2_dev_data,
        phase3_dev_corpus,
        save_dir=args.save_dir,
        target_languages=["aym", "bzd", "cni", "gn", "hch", "nah", "oto", "quy", "tar", "shp", "ctp"]
    )