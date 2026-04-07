import os

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys

sys.path.append(HOME_PATH)

from transformers import EvollaProcessor, EvollaForProteinText2Text

import json
import torch
import torch.distributed as dist


from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


class InferenceDataset(Dataset):
    def __init__(self, input_tsv, output_tsv=None):
        self.input_tsv = input_tsv
        existed_output = None
        if output_tsv is not None:
            existed_output = []
            with open(output_tsv, "r") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    uniprot_id, _, _, _, question = line.strip().split("\t")
                    question = json.loads(question)
                    existed_output.append((uniprot_id, question))

        self.inputs = []
        total_count = 0
        skipped_count = 0
        with open(input_tsv, "r") as f:
            for line in f:
                uniprot_id, aa_seq, foldseek, question = line.strip().split(
                    "\t"
                )
                if (
                    existed_output is not None
                    and (uniprot_id, question) in existed_output
                ):
                    skipped_count += 1
                    continue
                self.inputs.append((uniprot_id, aa_seq, foldseek, question))
                total_count += 1
        print(
            f"Total count: {total_count}, skipped count: {skipped_count} in InferenceDataset",
            flush=True,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        uniprot_id, aa_seq, foldseek, question = self.inputs[idx]
        return uniprot_id, aa_seq, foldseek, question

    def collate_fn(self, batch):
        uniprot_ids, aa_seqs, foldseeks, questions = zip(*batch)
        return uniprot_ids, aa_seqs, foldseeks, questions


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_tsv", type=str, required=True, help="Path to input tsv file."
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        required=True,
        help="Path to output tsv file.",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to model directory."
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Path to tokenizer directory.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--greedy_search",
        action="store_true",
        help="Use greedy search instead of sampling.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()
    return args


def main(args):
    input_tsv = args.input_tsv
    model_dir = args.model_dir
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir else model_dir
    output_tsv = args.output_tsv
    temperature = args.temperature
    greedy_search = args.greedy_search
    max_new_tokens = args.max_new_tokens

    # add function if output_tsv exists and generated half way
    if os.path.exists(output_tsv) and os.path.getsize(output_tsv) > 0:
        dataset = InferenceDataset(input_tsv, output_tsv=output_tsv)
    else:
        dataset = InferenceDataset(input_tsv)

    sampler = DistributedSampler(
        dataset, shuffle=False, rank=dist.get_rank()
    )  # rank 参数可省略
    data_loader = DataLoader(
        dataset, batch_size=1, sampler=sampler, collate_fn=dataset.collate_fn
    )

    hf_model = EvollaForProteinText2Text.from_pretrained(
        model_dir, device_map={"": dist.get_rank()}
    ).eval()
    # modify generation config
    generation_config = hf_model.generation_config
    if greedy_search:
        generation_config.do_sample = False
        if temperature is not None:
            print("Warning: temperature is ignored when using greedy search.")
    if max_new_tokens is not None:
        generation_config.max_new_tokens = max_new_tokens
        print(f"Setting max_new_tokens to {max_new_tokens}")
    if temperature is not None:
        generation_config.temperature = temperature
    hf_model.generation_config = generation_config

    print(f"Generation config: {hf_model.generation_config}")

    processor = EvollaProcessor.from_pretrained(tokenizer_dir)
    with open(output_tsv + str(dist.get_rank()), "w") as output_file:
        output_file.write(
            "seq_name\tcomment_type\tpred_sentence\tanswer_sentence\tquestion_sentence\n"
        )
        for i, batch in enumerate(tqdm(data_loader)):
            uniprot_ids, aa_seqs, foldseeks, questions = batch
            protein_informations = [
                {
                    "aa_seq": aa_seq,
                    "foldseek": foldseek,
                }
                for aa_seq, foldseek in zip(aa_seqs, foldseeks)
            ]
            messages_list = [
                [
                    {
                        "role": "system",
                        "content": "You are an AI expert that can answer any questions about protein.",
                    },
                    {"role": "user", "content": question},
                ]
                for question in questions
            ]
            input_dict = processor(
                protein_informations, messages_list, return_tensors="pt"
            ).to(hf_model.device)

            with torch.no_grad():
                # generated_ids = hf_model.generate(**input_dict, max_new_tokens=20)
                generated_ids = hf_model.generate(**input_dict)
            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for uniprot_id, generated_text, question in zip(
                uniprot_ids, generated_texts, questions
            ):
                generated_text = "".join(
                    generated_text.split("assistant\n\n")[1:]
                )
                output_file.write(
                    f"{uniprot_id}\thigh_quality\t{json.dumps(generated_text)}\t{json.dumps('no answer here')}\t{json.dumps(question)}\n"
                )
            output_file.flush()

    return


if __name__ == "__main__":
    args = parse_args()
    dist.init_process_group(backend="nccl")
    main(args)
    dist.destroy_process_group()
