import os
import argparse
import json
import random

parser = argparse.ArgumentParser(description="Script to shuffle line-by-line corpora for input to NLLB models")
parser.add_argument("--config", type=str, required=True, help="Shuffle config file (JSON)")

args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.load(config_file)

output_dir = config["corpus_file"] + "-shuffled"

if os.path.exists(output_dir):
    raise Exception("Corpus already shuffled--rename/delete the previous output directory to retry")
else:
    os.mkdir(output_dir)

test = []
train = []
dev = []

with open(config["corpus_file"]) as corpus_file:
    for i, line in enumerate(corpus_file):
        if i < config["test_size"]:
            test.append(line)
        elif i < config["test_size"] + config["dev_size"]:
            dev.append(line)
        else:
            train.append(line)

for i in config["shuffles"]:
    if i["seed"] != 0:
        random.seed(i["seed"])
    else:
        random.seed()

    train_shuffled = train.copy()
    random.shuffle(train_shuffled)

    with open(os.path.join(output_dir, "shuffle_" + i["id"] + "_" + config["corpus_file"]), "w") as out_file:
        for line in test:
            out_file.write(line)
        for line in dev:
            out_file.write(line)
        for line in train_shuffled:
            out_file.write(line)