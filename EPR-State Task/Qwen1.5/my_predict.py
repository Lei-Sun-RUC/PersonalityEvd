# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: predict
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/12/6 20:41
"""
    文件说明：
            
"""
import sys
sys.path.append('/data6/sl/EMNLP_pers_evidence/experiments/state_experiments')
import argparse
import torch
from peft import PeftModel
from data_formatter import DataFormatter
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from tqdm import tqdm
import time
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
 

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dlg_path", default="", type=str, help="")
    parser.add_argument("--state_anno_path", default="", type=str, help="")
    parser.add_argument("--data_type", default="state", type=str, help="which level data to use")
    parser.add_argument("--date", default="", type=str, help="")
    parser.add_argument("--datestr", default="", type=str, help="")
    parser.add_argument("--setting", default="lora", type=str, help="")
    parser.add_argument("--epoch", default=0, type=int, help="")
    parser.add_argument("--data_ratio", type=float, default=1, help="the train set ratio to be used")
    
    # Model
    parser.add_argument('--ori_model_dir', default="Baichuan-7B", type=str, help='')
    parser.add_argument('--model_dir', default="", type=str, help='')
    parser.add_argument("--device", type=str, default="0", help="")
    parser.add_argument("--mode", type=str, default="glm3", help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--do_sample", type=bool, default=False, help="")
    parser.add_argument("--top_p", type=float, default=0.01, help="")
    parser.add_argument("--temperature", type=float, default=0.01, help="")
    return parser.parse_args()


def predict_one_sample(input, model, tokenizer):
    messages = []
    messages.append({"role": "user", "content": input})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to("cuda:{}".format(args.device))

    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response


if __name__ == '__main__':
    args = parse_args()
    print(args)

        
    tokenizer = AutoTokenizer.from_pretrained(args.ori_model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.ori_model_dir, device_map="cuda:{}".format(args.device), torch_dtype=torch.float16, trust_remote_code=True)

    if args.setting=="lora":
        model = PeftModel.from_pretrained(model, args.model_dir, torch_dtype=torch.float16)
        print("Lora Model loaded!")
    elif args.setting=="zero-shot":
        print("No Lora Model loaded!")

    model.generation_config = GenerationConfig.from_pretrained(args.ori_model_dir,max_length=args.max_length, do_sample=args.do_sample,
                           top_p=args.top_p, temperature=args.temperature,max_new_tokens=512)
    print(model.generation_config)
   
    data_formatter=DataFormatter(args.dlg_path,args.state_anno_path,args.data_type)
    test_data=data_formatter.convert2qwen()
    pres=[[],[],[],[],[]]
    labels=[[],[],[],[],[]]

    
    data_split=args.state_anno_path.split('/')[-1].split('_')[0]
    result_file_dir=os.path.join("./results",f"{data_split}_{args.date}_{args.data_type}_data_ratio_{args.data_ratio}")
    if not os.path.exists(result_file_dir):
        os.mkdir(result_file_dir)
    result_file_path=os.path.join(result_file_dir,f"epoch_{args.epoch}-{args.datestr}.json")
    jsonl_file=jsonlines.open(result_file_path,"w")


    sta_time=time.time()
    #test_data=test_data[90:]
    for sample in tqdm(test_data):
        r = predict_one_sample(sample["input"], model, tokenizer)
        """
        print(sample["target_dim"]," level:",sample["level"])
        print("pred:",r+'\n',"label:",sample["output"])
        print()
        """
        save_data={"dia_id":sample["dia_id"],
                "target_speaker":sample["target_speaker"],
                "target_dim":sample["target_dim"],
                "level":sample["level"],
                "target_utt_ids":sample["target_utt_ids"],
                "label":sample["output"],
                "pre":r}
        jsonlines.Writer.write(jsonl_file,save_data) # 每行写入一个dict



