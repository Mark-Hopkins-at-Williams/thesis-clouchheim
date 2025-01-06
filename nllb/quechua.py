from torch.utils.data import DataLoader
from datasets import load_dataset

quechua_data = load_dataset("Llamacha/monolingual-quechua-iic")
#loader = DataLoader(dataset['train'], batch_size=4, shuffle=True)

def get_training_corpus():
    dataset = quechua_data["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]