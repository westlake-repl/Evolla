import os
import sys
HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(HOME_PATH)

import argparse
import json

import traceback
from threading import Thread
from utils.others import setup_seed, load_config, load_model_from_config

from transformers import TextIteratorStreamer

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--input_path", type=str, required=True, help="Path to the input file, each line is a tab-separated triplet of Uniprot ID (or any other identifier), sequence, foldseek, and question (in JSON format)")
args = parser.parse_args()
CONFIG_PATH = args.config_path
input_path = args.input_path

config = load_config(CONFIG_PATH)

if config.setting.seed:
    setup_seed(config.setting.seed)


model = load_model_from_config(config, local_rank=0, dtype="bf16")

with open(input_path, "r") as f:
    for line in f:
        line = line.strip()
        uniprot_id, sequence, foldseek, question = line.split("\t")
        question = json.loads(question)
        streamer = TextIteratorStreamer(
            model.llm_tokenizer,
            # skip_prompt=True,
            skip_prompt=False,
            skip_special_tokens=True,
        )
        
        mixed_sequence = "".join([s+f for s, f in zip(sequence, foldseek)])
        print(f"{uniprot_id}")
        print(f"{question}")
        print(f"{mixed_sequence}")
        generation_kwargs = { 
            "seqs": [mixed_sequence],
            "foldseeks": [None],
            "questions": [question],
            "streamer": streamer,
        }
        
        def generate_wrapper():
            try:
                model.generate(**generation_kwargs, **model.generate_config)
            except Exception as e:
                # traceback the exception
                traceback.print_exc()
                print(f"Exception in generate_wrapper: {e}")

        thread = Thread(target=generate_wrapper)
        thread.start()
        for a in streamer:
            print(a, end="", flush=True)
        thread.join()
        print("=" * 50)
