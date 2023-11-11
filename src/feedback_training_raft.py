#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""Alignment tuning example, such as RLHF."""

import os
import re
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, pipeline, AutoTokenizer

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)
from comet import load_from_checkpoint, download_model
from lmflow.datasets.dataset import Dataset
from lmflow.models.auto_model import AutoModel
from lmflow.pipeline.auto_pipeline import AutoPipeline

@dataclass
class RewardArguments:
    reward_type: Optional[str] = field(
        default="hf_pipeline",
        metadata={
            "help": (
                "type of reward model, support huggingface pipeline. Will"
                " support \"customized\" torch.nn.modules in the future."
            ),
        },
    )
    reward_model_or_path: Optional[str] = field(
        default="weqweasdas/hh_rlhf_rm",
        metadata={
            "help": (
                "reward model name (huggingface) or its path"
            ),
        },
    )
    reward_task: Optional[str] = field(
        default="sentiment-analysis",
        metadata={
            "help": "type of reward task, such as sentiment-analysis, detoxic."
        },
    )
    reward_model_args: Optional[str] = field(
        default="return_all_scores=True, function_to_apply=\"none\", batch_size=1",
        metadata={
            "help": (
                "extra arguments required by different type of reward models."
            ),
        },
    )

@dataclass
class FilterArguments:
    lang_detect: Optional[bool] = field(
        default=False,
        metadata={"help": "filter sample with wrong langauge"}
    )

    len_ratio: Optional[json.loads] = field(
        default=None,
        metadata={"help": "filter sample with abnormal length ratio"}
    )

    repeat_detect: Optional[bool] = field(
        default=False,
        metadata={"help": "filter sample with repeating error"}
    )

@dataclass
class GenArguments:
    temperature: Optional[float] = field(
        default=0.85,
        metadata={"help": "temperature of sampling"}
    )

def get_reward_function(reward_args, pipeline_args):
    args = reward_args
    reward_type = args.reward_type

    if reward_type == "hf_pipeline":

        # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
        # only for this model.
        rm_tokenizer = AutoTokenizer.from_pretrained(reward_args.reward_model_or_path)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id
        rm_tokenizer.padding_side = "left"
        
        hf_pipe = pipeline(
            reward_args.reward_task,
            model=reward_args.reward_model_or_path,
            device=f"cuda:{pipeline_args.local_rank}",
            tokenizer=rm_tokenizer
        )
        def reward_func(dataset: Dataset):
            if dataset.type != "text_only":
                raise NotImplementedError(
                    "reward function only accept \"text_only\" datasets"
                )
            pipe_kwargs = {
                "return_all_scores": True,
                "function_to_apply": "none",
                "batch_size": 1
            }

            data_dict = dataset.to_dict()
            texts_for_rewards = [
                sample["text"] for sample in data_dict["instances"]
            ]
            pipe_outputs = hf_pipe(texts_for_rewards, **pipe_kwargs)
            rewards = [output[0]["score"] for output in pipe_outputs]

            reward_dataset = Dataset.create_from_dict({
                "type": "float_only",
                "instances": [
                    { "value": reward } for reward in rewards
                ]
            })
            return reward_dataset

        return reward_func
    else:
        model_path = download_model(reward_args.reward_model_or_path)
        comet_model = load_from_checkpoint(model_path)
        comet_model.eval()
        comet_model.to(f"cuda:{pipeline_args.local_rank}")

        def reward_func(dataset: Dataset):
            data_dict = dataset.to_dict()
            input_lst = [
                re.search(r'### Input:(.*?)### Response:', e["text"], re.DOTALL).group(1).strip()
                for e in data_dict["instances"]
            ]
            output_lst = [
                re.search(r'### Response:(.*)', e["text"], re.DOTALL).group(1).strip()
                for e in data_dict["instances"]
            ]
            data = [
                {
                    "mt": o.strip(),
                    "src": s.strip(),
                    "ref": None
                } for o, s in zip(output_lst, input_lst)
            ]
            model_output = comet_model.predict(data, progress_bar=False, devices=[int(pipeline_args.local_rank)])
            if "Unbabel/unite-mup" in reward_args.reward_model_or_path:
                scores = model_output.metadata.src_scores
            else:
                scores = model_output.scores

            reward_dataset = Dataset.create_from_dict({
                "type": "float_only",
                "instances": [
                    { "value": reward } for reward in scores
                ]
            })
            return reward_dataset

        return reward_func


def main():
	# Parses arguments
    pipeline_name = "raft_aligner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((
        ModelArguments,
        DatasetArguments,
        PipelineArguments,
        RewardArguments,
        FilterArguments,
        GenArguments
    ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args, reward_args, filter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, pipeline_args, reward_args, filter_args, gen_args = parser.parse_args_into_dataclasses()

    # Initializes pipeline, dataset and model for reward training
    aligner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
        filter_args=filter_args,
        gen_args=gen_args
    )
    dataset = Dataset(data_args)
    # model = AutoModel.get_model(model_args, tune_strategy="none", ds_config=pipeline_args.deepspeed)
    model = AutoModel.get_model(model_args)


    # Initializes reward function
    reward_function = get_reward_function(reward_args, pipeline_args)

    reward_model_args = ModelArguments(arch_type="text_regression")
    reward_model = AutoModel.get_model(reward_model_args)
    reward_model.register_inference_function(reward_function)

    # Aligns model with rewards
    aligned_model = aligner.align(
        model=model,
        dataset=dataset,
        reward_model=reward_model,
    )


if __name__ == '__main__':
    main()