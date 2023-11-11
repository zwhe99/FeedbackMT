import os
import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig

# Instruction language, default: 'en'
lang_instruction = {
    'en': {'de': "German", 'en': "English", 'ja': "Japanese", 'zh': "Chinese", 'uk': "Ukrainian", 'cs': "Czech"},
}

NLLB_CODE = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "de": "deu_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn"
}

# Special tokens in llama
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


# Read task instruction, fill in languages
def read_instruct(path, src, tgt, lang_ins="en"):
    source, target = lang_instruction[lang_ins][src], lang_instruction[lang_ins][tgt]
    ins_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip().replace("[SRC]", source).replace("[TGT]", target)
            ins_list.append(line)
    return ins_list


# Read input data for inference
def read_input(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
    return input_data


# Assembly instruction and input data, handle hints
def create_prompt(instruct, input_data, template="prompt_no_input"):
    if "###" in instruct:
        instruct, input_suffix = instruct.split("###")
        hint = "\n\n### Hint: {}".format(input_suffix)
    else:
        instruct =  instruct
        hint = ""
    if template == "prompt_input":
        list_data_dict = [{"instruction": instruct, "input": p.strip() + hint} for p in input_data]
        prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    else:
        list_data_dict = [{"instruction": "\n\n".join([instruct, p.strip() + hint]).strip(), "input": ""} for p in input_data]
        prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    return sources


# Post-process the output, extract translations
def post_process(text):
    text = text.split("### Response:")[1].strip()
    text = text.replace("\n", " ")
    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]

    substring = "Below is an instruction that describes a task"
    index = text.find(substring)
    if index != -1:
        return text[:index].strip()
    else:
        return text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, required=True, help='model name in the hub or local path')
    parser.add_argument('--model-type', type=str, default="causal", choices=["causal", "s2s"])
    parser.add_argument("--lora-weights", type=str, default=None, help="use lora")
    parser.add_argument('--inst-file', '-ins', type=str, default=None, help='instruction file')
    parser.add_argument('--input-file','-i', type=str, required=True, help='input file')
    parser.add_argument('--output-file','-o', type=str, required=True, help='output file')
    parser.add_argument('--lang-pair', '-lp', type=str, default='zh-en', help='language pair: zh-en, en-de')
    parser.add_argument('--search-algorithm', '-sa', type=str, default='beam', help='search algorithms: sample, beam, beam_sample')
    parser.add_argument('--batch', '-b', type=int, default=2, help='batch size')
    parser.add_argument('--beam', type=int, default=4, help='beam size')
    parser.add_argument('--template', '-tp', type=int, default=1, help='0: prompt_no_input, 1: prompt_input')
    parser.add_argument('--temperature', '-t', type=float, default=0.7, help='temperature: 0.7 for text generation')
    parser.add_argument('--num-return-sequences', type=int, default=1, help='the number of highest scoring beams that should be returned')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    print(args.output_file)
    model_name_or_path = args.model_name_or_path
    model_type = args.model_type
    lora_weights = args.lora_weights
    inst_file = args.inst_file
    input_file = args.input_file
    output_file = args.output_file
    lang_pair = args.lang_pair
    search = args.search_algorithm
    batch = args.batch
    temperature = args.temperature
    temp = args.template
    seed = args.seed
    beam = args.beam
    num_return_sequences = args.num_return_sequences
    srcl, tgtl = lang_pair.split('-')
    template = "prompt_input" if temp > 0 else "prompt_no_input"

    hyp_file = output_file + ".hyp"
    num_done = len(read_input(hyp_file))
    num_total = len(read_input(input_file)) * num_return_sequences
    if num_done == num_total:
        print(f"Done {lang_pair}: {input_file}")
        exit(0)

    # Load checkpoints
    if torch.cuda.device_count() == 8:
        device_map = {'model.embed_tokens': 7, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 2, 'model.layers.9': 2, 'model.layers.10': 2, 'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 3, 'model.layers.14': 3, 'model.layers.15': 3, 'model.layers.16': 3, 'model.layers.17': 3, 'model.layers.18': 4, 'model.layers.19': 4, 'model.layers.20': 4, 'model.layers.21': 4, 'model.layers.22': 4, 'model.layers.23': 5, 'model.layers.24': 5, 'model.layers.25': 5, 'model.layers.26': 5, 'model.layers.27': 5, 'model.layers.28': 6, 'model.layers.29': 6, 'model.layers.30': 6, 'model.layers.31': 6, 'model.norm': 6, 'lm_head': 6}
    else:
        device_map = "auto"
    
    if model_type == "causal":
        torch_type = torch.float16
        if "llama-2" in model_name_or_path.lower():
            torch_type = torch.bfloat16
            print("Using bf16")

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_type, device_map=device_map)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map=device_map)
    if lora_weights is not None and os.path.exists(lora_weights):
        model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    print(model.hf_device_map)

    # bloom uses only fast tokenize
    to_use_fast = False
    if "bloom" in model_name_or_path:
        to_use_fast = True

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast)
    if model_type == "causal": 
        tokenizer.padding_side = "left"
    else:
        tokenizer.src_lng = NLLB_CODE[srcl]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "llama-2" in model_name_or_path.lower():
        tokenizer.pad_token = "<unk>"
        print("tokenizer.pad_token = <unk>")

    # Setting `pad_token_id` to `eos_token_id` for open-end generation.
    gen_config = GenerationConfig(temperature=temperature,
                                  do_sample=True,
                                  num_beams=1,
                                  max_new_tokens=256,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token_id=tokenizer.eos_token_id,
                                  num_return_sequences=num_return_sequences
                                )

    if search == "beam":
        gen_config = GenerationConfig(temperature=temperature,
                                      num_beams=beam,
                                      max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.eos_token_id,
                                      num_return_sequences=num_return_sequences
                                      )

    if search == "beam_sample":
        gen_config = GenerationConfig(temperature=temperature,
                                      do_sample=True,
                                      num_beams=beam,
                                      max_new_tokens=256,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.eos_token_id,
                                      num_return_sequences=num_return_sequences
                                      )

    # Prepare input data
    if model_type == "causal":
        if inst_file is not None:
            instructs = read_instruct(inst_file, srcl, tgtl)
            instruct = instructs[0] if len(instructs) > 0 else ""
        else: # In case instruction file is missing, then use input as instruction
            instruct = ""
            template = "prompt_no_input"
        input_data = read_input(input_file)
        prompt = create_prompt(instruct, input_data, template)
    else:
        input_data = read_input(input_file)
        prompt = [d.strip() for d in input_data]

    assert num_done % num_return_sequences == 0
    prompt = prompt[int(num_done / num_return_sequences):]


    # Generate
    torch.manual_seed(seed)
    with tqdm(total=num_total, desc=f"{lang_pair}") as pbar:
        pbar.update(num_done)
        with open(output_file, 'a', encoding='utf-8') as fo, open(output_file+".hyp", 'a', encoding='utf-8') as fo2:
            for i in range(0, len(prompt), batch):
                p = prompt[i:i+batch]
                with torch.autocast("cuda"):
                    tokenized = tokenizer(p, padding=True, return_tensors="pt")
                    input_ids = tokenized.input_ids.cuda()
                    attn_mask = tokenized.attention_mask.cuda()
                    if model_type == "causal":
                        input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
                        attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
                        forced_bos_token_id = None
                    else:
                        forced_bos_token_id=tokenizer.lang_code_to_id[NLLB_CODE[tgtl]]

                    generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config, forced_bos_token_id=forced_bos_token_id)
                decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for dec in decoded_tokens:
                    print(dec, file=fo, flush=True)
                    print(post_process(dec) if model_type == "causal" else dec, file=fo2, flush=True)
                pbar.update(len(decoded_tokens))