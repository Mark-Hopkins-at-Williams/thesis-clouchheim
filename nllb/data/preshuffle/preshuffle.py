import os
import argparse
import json
import random

parser = argparse.ArgumentParser(description="Script to shuffle line-by-line corpora for input to NLLB models")
parser.add_argument("--config", type=str, required=True, help="Shuffle config file (JSON)")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shuffled corpora (must not exist)")
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.load(config_file)

if os.path.exists(args.output_dir):
    raise Exception("Delete the previous output directory to retry")
else:
    os.mkdir(args.output_dir)

test = {}
train = {}
dev = {}

for filename in config["parallel_files"]:
    test[filename] = []
    train[filename] = []
    dev[filename] = []
    with open(filename) as corpus_file:
        for i, line in enumerate(corpus_file):
            # NOTE that we're grabbing test and dev from the top of the file only
            if i < config["test_size"]:
                test[filename].append(line)
            elif i < config["test_size"] + config["dev_size"]:
                dev[filename].append(line)
            else:
                train[filename].append(line)

for i in config["shuffles"]:
    if i["seed"] != 0:
        random.seed(i["seed"])
    else:
        random.seed()

    train_zipped = list(zip(*train.values()))
    random.shuffle(train_zipped)
    train_shuffled = {filename: list(lines) for filename, lines in zip(train.keys(), zip(*train_zipped))}

    for filename in config["parallel_files"]:
        with open(os.path.join(args.output_dir, "shuffle_" + i["id"] + "_" + filename), "w") as out_file:
            for line in test[filename]:
                out_file.write(line)
            for line in dev[filename]:
                out_file.write(line)
            for line in train_shuffled[filename]:
                out_file.write(line)