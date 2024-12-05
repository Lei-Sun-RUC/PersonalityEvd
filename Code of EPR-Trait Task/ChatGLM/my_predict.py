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

import argparse
import torch
from model import MODE
from peft import PeftModel
from data_formatter import DataFormatter
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from tqdm import tqdm
import time
import jsonlines
import json

def read_json(read_path):
    assert read_path.split('.')[-1] == 'json'
    with open(read_path, 'r') as data_file:
        data = [json.loads(line) for line in data_file]
    return data


def save_json(save_path,data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path,'w') as file:
        json.dump(data,file,indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dlg_path", default="", type=str, help="")
    parser.add_argument("--state_anno_path", default="", type=str, help="")
    parser.add_argument("--trait_anno_path", default="", type=str, help="")
    parser.add_argument("--data_type", default="state", type=str, help="which level data to use")
    parser.add_argument("--fold_num", default="fold_1", type=str, help="which fold to be test set")
    parser.add_argument("--train_or_test", default="train", type=str, help="train or inference")

    parser.add_argument("--date", default="", type=str, help="")
    parser.add_argument("--datestr", default="", type=str, help="")
    parser.add_argument("--setting", default="lora", type=str, help="")
    parser.add_argument("--epoch", default=0, type=int, help="")
    
    # Model
    parser.add_argument('--ori_model_dir', default="ChatGLM2-6B", type=str, help='')
    parser.add_argument('--model_dir', default="output-glm2/epoch-2-step-3900/", type=str, help='')
    parser.add_argument("--device", type=str, default="0", help="")
    parser.add_argument("--mode", type=str, default="glm3", help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--do_sample", type=bool, default=False, help="")
    parser.add_argument("--top_p", type=float, default=0.01, help="")
    parser.add_argument("--temperature", type=float, default=0.01, help="")
    return parser.parse_args()


def predict_one_sample(input, model, tokenizer, args):
    result, _ = model.chat(tokenizer, input, max_length=args.max_length, do_sample=args.do_sample,
                           top_p=args.top_p, temperature=args.temperature)
    return result


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    model = MODE[args.mode]["model"].from_pretrained(args.ori_model_dir, device_map="cuda:{}".format(args.device),torch_dtype=torch.float16)
    if args.setting=="lora":
        model = PeftModel.from_pretrained(model, args.model_dir, torch_dtype=torch.float16)
        print("Lora Model loaded!")
    elif args.setting=="zero-shot":
        print("No Lora Model loaded!")
    #model = MODE[args.mode]["model"].from_pretrained(args.model_path, device_map="cuda:{}".format(args.device),
    #                                                 torch_dtype=torch.float16)
   
    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(args.ori_model_dir)
    

    data_formatter=DataFormatter(args.dlg_path,args.state_anno_path,args.trait_anno_path,args.data_type,args.fold_num,args.train_or_test)
    test_data=data_formatter.convert2chatglm3()
    pres=[[],[],[],[],[]]
    labels=[[],[],[],[],[]]

    result_file_dir=os.path.join("./results",f"{args.date}_{args.data_type}_{args.fold_num}")
    if not os.path.exists(result_file_dir):
        os.mkdir(result_file_dir)
    result_file_path=os.path.join(result_file_dir,f"epoch_{args.epoch}-{args.datestr}.json")
    jsonl_file=jsonlines.open(result_file_path,"w")

    

    sta_time=time.time()
    for sample in tqdm(test_data):
        r = predict_one_sample(sample["input"], model, tokenizer, args)
        
        
        save_data={"target_speaker":sample["target_speaker"],
                "target_dim":sample["target_dim"],
                "dialogue_num":sample["dialogue_num"],
                "level":sample["level"],
                "dlg_id":sample["dlg_id"],
                "label":sample["output"],
                "pre":r}
        jsonlines.Writer.write(jsonl_file,save_data) # 每行写入一个dict

    """ 
    if args.data_type=="state":
        test_data=read_json(result_file_path)
        
        chi2eng={
            "开放性":"openness",
            "尽责性":"conscientiousness",
            "外向性":"extraversion",
            "宜人性":"agreeableness",
            "神经质性":"neuroticism"
        }
        pres_pre_role={}
        for data in test_data:

            if data["target_speaker"] not in pres_pre_role:
                pres_pre_role[data["target_speaker"]]={
                'openness': {},
                'conscientiousness': {},
                'extraversion': {},
                'agreeableness': {},
                'neuroticism': {}
                }
            
            if "句子id：无" not in data["pre"] and "无法判断" not in data["pre"]:
                pres_pre_role[data["target_speaker"]][chi2eng[data["target_dim"]]][data["dia_id"]]=data["pre"]
        
        save_json(os.path.join(result_file_dir,f"epoch_{args.epoch}-{args.datestr}_pre_role.json"),pres_pre_role)
    """ 
    
