import json
import os
from bert_score import score
from sklearn.metrics import accuracy_score,f1_score
import re

def bert_score(cands,refs,gpu_id):

    P, R, F1 = score(cands, refs, model_type="/data2/sl/pretrain_models/bert-base-chinese", lang="zh", device=gpu_id, verbose=True)
    return F1, F1.mean()

def read_json(read_path):
    assert read_path.split('.')[-1] == 'json'
    with open(read_path, 'r') as data_file:
        data = [json.loads(line) for line in data_file]
    return data



def format_trait_data(data,type):
    # extract dia_id,nat_lang and trait_level
    # type: "label" or "pre"
    formatted_data={"dia_id":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]},
          "nat_lang":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]},
          "level":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]}}

    for sample in data:

        p1 = re.compile(r'[（](.*?)[）]', re.S)  #最小匹配

        # precess dia id
        dia_ids=re.findall(p1, sample[type])
        dia_ids='，'.join(dia_ids)
        dia_ids=dia_ids.replace(",","，")

        binary_ids=[0]*sample["dialogue_num"]
        
        string_ids=dia_ids.split('，')
        
        
        int_ids=[int(id) for id in string_ids if id!='']
        for id in range(sample["dialogue_num"]):
            if id+1 in int_ids:
                binary_ids[id]=1
        
        # process nat_lang
        print(sample[type])
       
        nat_lang_evi,level=sample[type].split('\n综上所述，')
        nat_lang_evi = re.sub('\（.*?\）', '', nat_lang_evi)
        nat_lang_evi=nat_lang_evi.strip()

        # process trait level
        if f"{sample['target_dim']}高" in level:
            level=f"{sample['target_dim']}高"
        elif f"{sample['target_dim']}低" in level:
            level=f"{sample['target_dim']}低"
        else:
            level="无法判断"


        formatted_data["dia_id"][sample['target_dim']].extend(binary_ids)
        formatted_data["nat_lang"][sample['target_dim']].append(nat_lang_evi)
        formatted_data["level"][sample['target_dim']].append(level)

    return formatted_data




def compute_trait_dia_level_metrics(read_dir):
    # compute dia_id_f1, nat_lang and level_acc

    file_names=os.listdir(read_dir)
    if "all_metrics.txt" in file_names:
        file_names.remove("all_metrics.txt")
    file_names.sort(key=lambda x:int(x.split('-')[0].split('_')[-1]))
    
    f=open(os.path.join(read_dir,"all_metrics.txt"),'w')

    for file_name in file_names:
        read_path=os.path.join(read_dir,file_name)

        infer_data=read_json(read_path)
        
        pres=format_trait_data(infer_data,"pre")
        labels=format_trait_data(infer_data,"label")


        f.writelines(f"{file_name}\n")
        sum_scores={"utt_id_f1":0,"nat_lang_score":0,"level_acc":0}
        for dim in ["开放性","尽责性","外向性","宜人性","神经质性"]:
            
            # utt id metric
            dia_id_f1 = f1_score(labels["dia_id"][dim],pres["dia_id"][dim])
            sum_scores["utt_id_f1"]+=dia_id_f1
            # nat_lang metric
            final_pres,final_labels=[],[]
            for i in range(len(pres["nat_lang"][dim])):
                sub_dim_pres=pres["nat_lang"][dim][i].split("\n")
                print(sub_dim_pres)
                if len(sub_dim_pres)>3:
                    sub_dim_pres=sub_dim_pres[:3]
                assert len(sub_dim_pres)==3
                final_pres+=sub_dim_pres

                sub_dim_labels=labels["nat_lang"][dim][i].split("\n")
                assert len(sub_dim_labels)==3
                final_labels+=sub_dim_labels
            
            _,nat_lang_score = bert_score(final_pres,final_labels,gpu_id=7)
            sum_scores["nat_lang_score"]+=nat_lang_score

            # level metric
            level_acc=accuracy_score(labels["level"][dim],pres["level"][dim])
            sum_scores["level_acc"]+=level_acc

            f.writelines(f"{dim} dia_id_f1 : {dia_id_f1} nat_lang_score : {nat_lang_score} level_acc : {level_acc}\n")

            if dim=="神经质性":
                avg_utt_id_f1=sum_scores["utt_id_f1"]/5
                avg_nat_lang_score=sum_scores["nat_lang_score"]/5
                avg_level_acc=sum_scores["level_acc"]/5
                f.writelines(f"avg_utt_id_f1 : {avg_utt_id_f1} avg_nat_lang_score : {avg_nat_lang_score} avg_level_acc : {avg_level_acc}\n\n")

def format_level(data):
    # extract level

    formatted_data={"pres":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]},
          "labels":{"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]}}
    
    for sample in data:
        
        print(sample["pre"])
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
                f.writelines(f"avg_level_acc : {sum_level_acc/5}\n\n")



def compute_trait_nat_lang_metrics(read_dir):
    # compute dia_id_f1 and level_acc

    file_names=os.listdir(read_dir)
    if "all_metrics.txt" in file_names:
        file_names.remove("all_metrics.txt")
    file_names.sort(key=lambda x:int(x.split('-')[0].split('_')[-1]))
    
    f=open(os.path.join(read_dir,"all_metrics.txt"),'w')

    for file_name in file_names:
        read_path=os.path.join(read_dir,file_name)

        infer_data=read_json(read_path)
        
        pres=format_trait_data(infer_data,"pre")
        labels=format_trait_data(infer_data,"label")


        f.writelines(f"{file_name}\n")
        sum_scores={"nat_lang_score":0}
        for dim in ["开放性","尽责性","外向性","宜人性","神经质性"]:
            
            final_pres,final_labels=[],[]
            for i in range(len(pres["nat_lang"][dim])):
                sub_dim_pres=pres["nat_lang"][dim][i].split("\n")
                assert len(sub_dim_pres)==3
                final_pres+=pres["nat_lang"][dim][i].split("\n")

                sub_dim_labels=labels["nat_lang"][dim][i].split("\n")
                assert len(sub_dim_labels)==3
                final_labels+=sub_dim_labels
            
            #nat_lang metric
            _,nat_lang_score = bert_score(final_pres,final_labels,gpu_id=7)
            #print(f"{dim} nat_lang_score : {nat_lang_score}")
            f.writelines(f"{dim} nat_lang_score : {nat_lang_score}\n")
            sum_scores["nat_lang_score"]+=nat_lang_score

            if dim=="神经质性":
                avg_nat_lang_score=sum_scores["nat_lang_score"]/5
                
                f.writelines(f"avg_nat_lang_score : {avg_nat_lang_score}\n\n")

               
def compute_trait_nat_lang_LLM_eval(read_path):

    llm_eval_results=read_json(read_path)
    BF_scores={"开放性":[], "尽责性":[], "外向性":[], "宜人性":[], "神经质性":[]}
    for result in llm_eval_results:
      
        #print(result["score"])
        print(result["score"])
        scores=result["score"].split(',')
        scores=[int(score) for score in scores]

        
        BF_scores[result["target_dim"]].append(sum(scores)/len(scores))

    

    for dim in ["开放性","尽责性","外向性","宜人性","神经质性"]:
        print(dim,sum(BF_scores[dim])/len(BF_scores[dim]))
            


if __name__ == "__main__":
 
    pass

