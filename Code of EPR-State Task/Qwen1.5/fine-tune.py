import sys
sys.path.append('/data6/sl/EMNLP_pers_evidence/experiments/state_experiments')
from data_formatter import DataFormatter
import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments
from transformers import DataCollatorForSeq2Seq


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen1.5-7B-Chat")


@dataclass
class DataArguments:
    dlg_path: str = field(
        default=None, metadata={"help": "Path to the dialogue data."}
    )
    state_anno_path: str = field(
        default=None, metadata={"help": "Path to the state anno data."}
    )
    
    data_type: str = field(
        default="level", metadata={"help": "data_type"}
    )
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    lora_dim: int = field(default=16)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.1)
   


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_info,
        tokenizer,
        model_max_length,
    ):
        super(SupervisedDataset, self).__init__()
        
        data_formatter=DataFormatter(data_info["dlg_path"],data_info["state_anno_path"],data_info["data_type"])
        self.data=data_formatter.convert2qwen()



        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        
    

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):

        input_ids, attention_mask, labels = [], [], []
       
        instruction = self.tokenizer(f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]  
        if len(input_ids) > self.model_max_length:  # 做一个截断
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            labels = labels[:self.model_max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(data_args)
    print(training_args)
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=training_args.lora_dim,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    data_info = {
            "dlg_path":data_args.dlg_path,
            "state_anno_path":data_args.state_anno_path,
            "data_type":data_args.data_type,}
    dataset = SupervisedDataset(
        data_info, tokenizer, training_args.model_max_length
    )
    
    print("len(dataset)",len(dataset))
    
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    
if __name__ == "__main__":
    train()
    