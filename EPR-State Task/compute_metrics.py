import json
import os
from bert_score import score
from sklearn.metrics import accuracy_score,f1_score
import re
import jsonlines

def bert_score(cands,refs,gpu_id):

    P, R, F1 = score(cands, refs, model_type="/data2/sl/pretrain_models/bert-base-chinese", lang="zh", device=gpu_id, verbose=True)
    return F1, F1.mean()

#f1,avg_f1=bert_score(["乔卫东面对宋倩的阻拦和不满并未生气。这体现了乔卫东情绪稳定，不易生气。","我喜欢你"],["在这段对话中，没有体现乔卫东神经质性特征的句子。","我不喜欢你"],gpu_id=1)
#print(f1,avg_f1)



def read_json(read_path):
    assert read_path.split('.')[-1] == 'json'
    with open(read_path, 'r') as data_file:
        data = [json.loads(line) for line in data_file]
    return data

def format_evi_level(data,type):
    # extract utt_id,nat_lang and level
    # type: "label" or "pre"
    formatted_data={"utt_id":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]},
          "nat_lang":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]},
          "level":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]}}
    

    for sample in data:
        print(sample[type])
        evi,level=sample[type].split('因此，')
        binary_ids=[0]*len(sample["target_utt_ids"])
        if "无法判断" in level:
            
            # precess utt id
            
            # process nat_lang
            nat_lang=evi.replace('在这段对话中，','')

            # process level
            level="无法判断"
        else:
            # precess utt id and nat_lang
            #print(evi)
            utt_ids,nat_lang=evi.split('句，')
            utt_ids_list = re.findall("\d+\.?\d*", utt_ids)
            utt_ids_list = list(map(int,utt_ids_list))

            for id,utt_id in enumerate(sample["target_utt_ids"]):
                if utt_id in utt_ids_list:
                    binary_ids[id]=1

            # process level
            if f"{sample['target_dim']}高" in level:
                level=f"{sample['target_dim']}高"
            elif f"{sample['target_dim']}低" in level:
                level=f"{sample['target_dim']}低"

        formatted_data["utt_id"][sample['target_dim']].extend(binary_ids)
        formatted_data["nat_lang"][sample['target_dim']].append(nat_lang)
        formatted_data["level"][sample['target_dim']].append(level)

    return formatted_data


def compute_evi_level_metrics(read_dir):

    file_names=os.listdir(read_dir)
    if "all_metrics.txt" in file_names:
        file_names.remove("all_metrics.txt")
    file_names.sort(key=lambda x:int(x.split('-')[0].split('_')[-1]))
    
    f=open(os.path.join(read_dir,"all_metrics.txt"),'w')

    for file_name in file_names:
        read_path=os.path.join(read_dir,file_name)

        infer_data=read_json(read_path)

        pres=format_evi_level(infer_data,"pre")
        #print(pres["utt_id"]["开放性"])
        labels=format_evi_level(infer_data,"label")

       
        f.writelines(f"{file_name}\n")
        sum_scores={"utt_id_f1":0,"nat_lang_score":0,"level_acc":0}
        for dim in ["开放性","尽责性","外向性","宜人性","神经质性"]:
            
            # utt id metric
            utt_id_f1 = f1_score(labels["utt_id"][dim],pres["utt_id"][dim],average="weighted")
            sum_scores["utt_id_f1"]+=utt_id_f1
            # nat_lang metric
            _,nat_lang_score = bert_score(pres["nat_lang"][dim],labels["nat_lang"][dim],gpu_id=7)
            sum_scores["nat_lang_score"]+=nat_lang_score
            # level metric
            level_acc=accuracy_score(labels["level"][dim],pres["level"][dim])
            sum_scores["level_acc"]+=level_acc
            
            f.writelines(f"{dim} utt_id_f1 : {utt_id_f1} nat_lang_score : {nat_lang_score} level_acc : {level_acc}\n")
            if dim=="神经质性":
                avg_utt_id_f1=sum_scores["utt_id_f1"]/5
                avg_nat_lang_score=sum_scores["nat_lang_score"]/5
                avg_level_acc=sum_scores["level_acc"]/5
                avg_scores=(avg_utt_id_f1+avg_nat_lang_score+avg_level_acc)/3
                f.writelines(f"avg_utt_id_f1 : {avg_utt_id_f1} avg_nat_lang_score : {avg_nat_lang_score} avg_level_acc : {avg_level_acc} avg_scores : {avg_scores}\n")
                f.writelines("\n")


def format_level(data):
    # extract utt_id,nat_lang and level

    formatted_data={"pres":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]},
          "labels":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]}}
    
    for sample in data:
        
        #print(sample["pre"])
        assert sample["pre"] in [f"{sample['target_dim']}高",f"{sample['target_dim']}低","无法判断"]
        assert sample["label"] in [f"{sample['target_dim']}高",f"{sample['target_dim']}低","无法判断"]
        formatted_data["pres"][sample['target_dim']].append(sample["pre"])
        formatted_data["labels"][sample['target_dim']].append(sample["label"])

    return formatted_data



def compute_level_metrics(read_dir):

    file_names=os.listdir(read_dir)
    if "all_metrics.txt" in file_names:
        file_names.remove("all_metrics.txt")
    file_names.sort(key=lambda x:int(x.split('-')[0].split('_')[-1]))
    
    f=open(os.path.join(read_dir,"all_metrics.txt"),'w')

    for file_name in file_names:
        read_path=os.path.join(read_dir,file_name)

        infer_data=read_json(read_path)

        formatted_data=format_level(infer_data)

  
        f.writelines(f"{file_name}\n")
        sum_level_acc=0
        for dim in ["开放性","尽责性","外向性","宜人性","神经质性"]:
            
            # level metric
            level_acc=accuracy_score(formatted_data["labels"][dim],formatted_data["pres"][dim])
            sum_level_acc+=level_acc
            f.writelines(f"{dim} level_acc : {level_acc}\n")
            if dim=="神经质性":
                f.writelines(f"avg_level_acc : {sum_level_acc/5}\n")
                f.writelines("\n")
               


def compute_nat_lang_LLM_eval(read_path):

    llm_eval_results=read_json(read_path)
    BF_scores={"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]}
    for result in llm_eval_results:
        if result["score"]=="You have exceeded the maximum number of requests for the free trial.":
            continue
        #print(result["score"])
        score=result["score"].replace("\n","")
        score=int(score)
        BF_scores[result["target_dim"]].append(score)

    

    for dim in ["开放性","尽责性","外向性","宜人性","神经质性"]:
        print(dim,sum(BF_scores[dim])/len(BF_scores[dim]))
            
           



if __name__ == "__main__":

    #compute_level_metrics("./ChatGLM/results/valid_20240521_level_data_ratio_1.0")
    compute_evi_level_metrics("/data6/sl/EMNLP_pers_evidence/experiment/state_experiments/ChatGLM/results/test_20240604_both_data_ratio_1.0/COT-FT")
    #compute_level_metrics("/data6/sl/EMNLP_pers_evidence/experiment/state_experiments/ChatGLM/results/test_20240604_both_data_ratio_1.0/Direct-FT")
    #compute_level_metrics("./Baichuan2/results/valid_20240517_level")
    #compute_level_metrics("/data6/sl/EMNLP_pers_evidence/experiment/state_experiments/Baichuan2/results/test_20240521_level")

    #compute_nat_lang_LLM_eval("/data6/sl/EMNLP_pers_evidence/experiment/state_experiments/Qwen1.5/results/test_20240604_evi_level_data_ratio_1/nat_lang_GPT4_eval/final_results.json")

    

    

