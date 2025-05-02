import os
import pandas as pd
import random
from tqdm import tqdm
from multilingualdata import StreamingBitext

def split_moses_files(src_file, tgt_file, train_ratio=0.9):
    with open(src_file, 'r', encoding='utf-8') as src, open(tgt_file, 'r', encoding='utf-8') as tgt:
        src_lines = src.readlines()
        tgt_lines = tgt.readlines()
    
    assert len(src_lines) == len(tgt_lines), "Source and target files must have the same number of lines."
    
    # Combine and shuffle pairs
    sentence_pairs = list(zip(src_lines, tgt_lines))
    random.shuffle(sentence_pairs)
    
    # Split into train and dev sets
    split_idx = int(len(sentence_pairs) * train_ratio)
    train_pairs = sentence_pairs[:split_idx]
    dev_pairs = sentence_pairs[split_idx:]
    
    # Save the training and dev data to new files
    def save_split(pairs, src_output, tgt_output):
        with open(src_output, 'w', encoding='utf-8') as src_out, open(tgt_output, 'w', encoding='utf-8') as tgt_out:
            for src_line, tgt_line in pairs:
                src_out.write(src_line)
                tgt_out.write(tgt_line)

    save_split(train_pairs, 'opus_data/train.en', 'opus_data/train.es')
    save_split(dev_pairs, 'opus_data/dev.en', 'opus_data/dev.es')

# Example usage
split_moses_files('opus_data/opus_data/OpenSubtitles.en-es.en', 'opus_data/opus_data/OpenSubtitles.en-es.es', train_ratio=0.9)

train_bitext = StreamingBitext('eng_Latn', 'spa_Latn', 'opus_data/train.en', 'opus_data/train.es')
dev_bitext = StreamingBitext('eng_Latn', 'spa_Latn', 'opus_data/dev.en', 'opus_data/dev.es')