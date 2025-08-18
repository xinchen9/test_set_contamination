import os
import math
import random
import json
import queue
import numpy as np
from scipy.stats import t as tdist

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM

from multiprocessing import Process, Queue
from tqdm import tqdm
import fire

os.environ['TOKENIZERS_PARALLELISM'] = "True"

flatten = lambda l: [x for s in l for x in s]
shuffle = lambda l: random.sample(l, k=len(l))

def load_dataset(dataset_path):
    # For loading a JSON-serialized list of examples.
    if dataset_path.endswith(".json"):
        print("loading from json...")
        with open(dataset_path, "r") as f:
            data = f.read()
            examples = json.loads(data)
            return examples
    # For loading a dataset where each example is on its own line.
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    return lines


def main(model_name_or_path,
         dataset_path,
         context_len=2048,
         stride=1024,
         num_shards=50,
         permutations_per_shard=250,
         random_seed=0,
         log_file_path=None,
         max_examples=None):
    
    mp.set_start_method("spawn", force=True)

    #set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)

    #load the dataset
    examples = load_dataset(dataset_path)
    num_examples = len(examples)
    if max_examples is not None:
        if num_examples < max_examples:
            max_examples = num_examples
        examples = examples[:max_examples]

        print(f"Load {max_examples} examples of {num_examples} from {dataset_path}")
    else:
        print(f"Load {num_examples} of all dataset {dataset_path}")

    #load tokeizer and tokenize the examples
    tkn = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_examples = [tkn.encode(ex) for ex in examples]
    print("Test runing...")


if __name__ == '__main__':
    # Optional: make CUDA errors synchronous for debugging
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Optional: use device-side assertions if your PyTorch build supports it
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    fire.Fire(main)