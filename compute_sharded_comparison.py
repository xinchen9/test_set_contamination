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

def worker(model_name_or_path,
           context_len,
           stride,
           local_device_index,   # MUST be 0..(visible_gpus-1)
           main_queue,
           worker_queue):
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in worker")
        
        visible = torch.cuda.device_count()
        if not (0 <= local_device_index < visible):
            raise RuntimeError(f"Worker sees {visible} GPU(s); requested local {local_device_index}")

        # Set current device BEFORE any CUDA ops
        torch.cuda.set_device(local_device_index)
        device_str = f"cuda:{local_device_index}"

        m = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        m.to(device_str)
        m.eval()

        while True:
            item = worker_queue.get()
            if item is None:
                break
        
    except Exception as e:
        # Report error to parent and exit cleanly
        main_queue.put(("error", f"PID {os.getpid()} failed: {repr(e)}"))
    finally:
        try:
            del m
            torch.cuda.empty_cache()
        except Exception:
            pass



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

    ctx = mp.get_context("spawn")

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

    #determine visiable GPUs (local indices 0 ...N-1)
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        raise RuntimeError("No GPUs bisible. Chenck CUDA, drives")

    num_workers = visible_gpus 
    print(f"Launching {num_workers} workes over {visible_gpus} visible GPU(s)")

    processes = []
    # main_queue = Queue()
    # worker_queues = [Queue() for _ in range(num_workers)]
    # Create queues from the SAME context
    main_queue = ctx.SimpleQueue()   # SimpleQueue avoids SemLock entirely
    worker_queues = [ctx.SimpleQueue() for _ in range(num_workers)]

    for local_idx in range(num_workers):
        p = ctx.Process(target=worker,
                    args=(model_name_or_path,
                          context_len,
                          stride,
                          local_idx,           # pass LOCAL index
                          main_queue,
                          worker_queues[local_idx]))
        
        p.start()
        processes.append(p)

        #wait untio each GPU load a model
        

    for wq in worker_queues:
        wq.put(None)
    for p in processes:
        p.join()

    print("Test runing...")


if __name__ == '__main__':
    # Optional: make CUDA errors synchronous for debugging
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Optional: use device-side assertions if your PyTorch build supports it
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    fire.Fire(main)