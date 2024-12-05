import json
from answer_example import example
from BigFive_definition import definition

def read_json(read_path):
    assert read_path.split('.')[-1] == 'json'
    with open(read_path, 'r') as f:
        data = json.load(f)
    return data


class DataFormatter:
    def __init__(self,dlg_path,state_anno_path,trait_anno_path,data_type,fold_num,train_or_test="train"):
        
        # param data_type: ["level", "utt_id", "analysis"]
        #  level: 只用label
        #  utt_id: 加上 evidence utterance id
        #  analysis: 加上自然语言解释

        assert data_type in ["evi_level", "level", "state_anno_trait_evi_level","both","state_anno"]
        self.data_type=data_type
        self.fold_num=fold_num
        self.train_or_test=train_or_test

        self.dialogue=read_json(dlg_path)
        self.state_anno_path=state_anno_path
        self.state_anno=read_json(state_anno_path)
        self.trait_anno=read_json(trait_anno_path)

        self.dim_eng2chi={"openness":"开放性",
        "conscientiousness":"尽责性",
        "extraversion":"外向性",
        "agreeableness":"宜人性",
        "neuroticism":"神经质性"}
        self.dim_chi2eng={"开放性":"openness",
                          "尽责性":"conscientiousness",
                          "外向性":"extraversion",
                          "宜人性":"agreeableness",
                          "神经质性":"neuroticism"}

        self.folds={
            "fold_1":["胡一菲", "曾小贤", "林宛瑜", "刘星", "邓小可母亲", "陈美嘉", "邓小可", "张伟", "夏东海", "吕子乔", "郑海潮", "关谷神奇", "陆展博", "刘梅", "唐悠悠", "童文洁", "彭永辉", "王胜男", "罗槟", "邱莹莹", "朱丽", "崔朕", "陆小贝", "赵小亮"],
            "fold_2":["福子", "陈可依", "童小麒", "鲍家明", "刘光耀", "王媛", "江达琳", "潘芸", "王柏川", "李三妹", "安迪", "苏明哲", "季胜利", "罗海燕", "毕胜男", "苏大强", "郑远东", "金志明", "曲筱绡", "樊胜美", "乔卫东", "福方树", "方圆", "乔英子"],
            "fold_3":["余峥", "罗素", "宋大楠", "吴非", "戴曦", "邹男", "苏明玉", "苏明成", "郝泽宇", "林妙妙", "张亮忠", "罗玥", "吴佳妮", "卫哲", "李闻雨", "夏峰", "于果", "姚澜", "齐大胜", "方一凡", "刘静", "林大为", "江焱", "宋倩"]
           }


 

    def trait_sample_with_GT_state(self):
        # 利用ground truth 的state标注 进行trait的预测

        trait_anno={}
        if self.train_or_test=="train":
            train_roles= self.dialogue.keys()-self.folds[self.fold_num]
            for role in train_roles:
                trait_anno[role]=self.trait_anno[role]

        elif  self.train_or_test=="test":
            for role in self.folds[self.fold_num]:
                trait_anno[role]=self.trait_anno[role]

        trait_data=[]
        for role_name,role_data in trait_anno.items():
            for dim,dim_anno in role_data.items():

                """
                dialogues_state_label=""
                role_dialogues=self.dialogue[role_name]
                dia_ids=dim_anno["dlg_id"].replace(";",",").replace("#",",").split(",")
                #print(role_name,dim,dia_ids)
             
                dia_ids=[int(dia_id) for dia_id in dia_ids if dia_id!='']
                dia_ids.sort()
                dia_ids=[str(dia_id) for dia_id in dia_ids ]
               
                #print(role_name,dim,dia_ids)
                """
                dialogues_state_label=""
                role_dialogues=self.dialogue[role_name]
                dia_ids=role_dialogues["dialogue"].keys()

                for dia_id in dia_ids:
                    
                    dialogues_state_label+=f"对话{dia_id}：\n"+'\n'.join(list(role_dialogues["dialogue"][dia_id]))+"\n"
                    dialogues_state_label+=f"对话{dia_id}的分析：{self.state_anno[role_name]['annotation'][dia_id][dim]['nat_lang']}\n\n"
                    

                sample_data={
                    "target_speaker":role_name,
                    "target_dim":self.dim_eng2chi[dim],
                    "dialogue_num":len(role_dialogues["dialogue"]),
                    "dialogues_state_label":dialogues_state_label,
                    "level":dim_anno["level"],
                    "dlg_id":dim_anno["dlg_id"],
                    "nat_lang":dim_anno["nat_lang"]
                }
                trait_data.append(sample_data)
        
        return trait_data
    

    

    def state_anno_only(self):
        # 只用state_anno 进行trait的预测
        trait_anno={}
        if self.train_or_test=="train":
            train_roles= self.dialogue.keys()-self.folds[self.fold_num]
            for role in train_roles:
                trait_anno[role]=self.trait_anno[role]

        elif  self.train_or_test=="test":
            for role in self.folds[self.fold_num]:
                trait_anno[role]=self.trait_anno[role]

        trait_data=[]
        for role_name,role_data in trait_anno.items():
            for dim,dim_anno in role_data.items():

              
                dialogues_state_label=""
                role_dialogues=self.dialogue[role_name]
                dia_ids=role_dialogues["dialogue"].keys()

                for dia_id in dia_ids:
                    
                    #dialogues_state_label+=f"对话{dia_id}：\n"+'\n'.join(list(role_dialogues["dialogue"][dia_id]))+"\n"
                    dialogues_state_label+=f"对话{dia_id}的分析：{self.state_anno[role_name]['annotation'][dia_id][dim]['nat_lang']}\n"
                    

                sample_data={
                    "target_speaker":role_name,
                    "target_dim":self.dim_eng2chi[dim],
                    "dialogue_num":len(role_dialogues["dialogue"]),
                    "dialogues_state_label":dialogues_state_label,
                    "level":dim_anno["level"],
                    "dlg_id":dim_anno["dlg_id"],
                    "nat_lang":dim_anno["nat_lang"]
                }
                trait_data.append(sample_data)
        
        return trait_data

    def trait_sample_only(self):
        # 利用预测的state 结果进行trait的预测，而不是ground truth 的state标注

        trait_anno={}
        if self.train_or_test=="train":
            train_roles= self.dialogue.keys()-self.folds[self.fold_num]
            for role in train_roles:
                trait_anno[role]=self.trait_anno[role]

        elif  self.train_or_test=="test":
            for role in self.folds[self.fold_num]:
                trait_anno[role]=self.trait_anno[role]

        trait_data=[]
        for role_name,role_data in trait_anno.items():
            for dim,dim_anno in role_data.items():

                dialogues=""
                role_dialogues=self.dialogue[role_name]
                dia_ids=role_dialogues["dialogue"].keys()
               

                for dia_id in dia_ids:     
                    dialogues+=f"对话{dia_id}：\n"+'\n'.join(list(role_dialogues["dialogue"][dia_id]))+"\n\n"
            
                sample_data={
                    "target_speaker":role_name,
                    "target_dim":self.dim_eng2chi[dim],
                    "dialogue_num":len(role_dialogues["dialogue"]),
                    "dialogues_state_label":dialogues,
                    "level":dim_anno["level"],
                    "dlg_id":dim_anno["dlg_id"],
                    "nat_lang":dim_anno["nat_lang"]
                }
                trait_data.append(sample_data)

        return trait_data   

    def convert2chatglm3(self):

        if self.data_type=="evi_level":
            trait_data=self.trait_sample_only()
        elif self.data_type=="level":
            trait_data=self.trait_sample_only()
        elif self.data_type=="both":
            trait_data=self.trait_sample_only()
        elif self.data_type=="state_anno_trait_evi_level":
            trait_data=self.trait_sample_with_GT_state()
        elif self.data_type=="state_anno":
            trait_data=self.state_anno_only()
        

        

        final_data=[]
        if self.data_type!="both":
            for sample_data in trait_data:
                target_speaker,target_dim,dialogue_num,level,dlg_id,input,output=self.get_input_output(sample_data,self.data_type)
                final_data.append({"target_speaker":target_speaker,"target_dim":target_dim,"dialogue_num":dialogue_num,"level":level,"dlg_id":dlg_id,"input":input,"output":output})
        else:
            for sample_data in trait_data:
                target_speaker,target_dim,dialogue_num,level,dlg_id,input,output=self.get_input_output(sample_data,"evi_level")
                final_data.append({"target_speaker":target_speaker,"target_dim":target_dim,"dialogue_num":dialogue_num,"level":level,"dlg_id":dlg_id,"input":input,"output":output})
                target_speaker,target_dim,dialogue_num,level,dlg_id,input,output=self.get_input_output(sample_data,"level")
                final_data.append({"target_speaker":target_speaker,"target_dim":target_dim,"dialogue_num":dialogue_num,"level":level,"dlg_id":dlg_id,"input":input,"output":output})

        return final_data

    def convert2qwen(self):
        
        if self.data_type=="evi_level":
            trait_data=self.trait_sample_only()
        elif self.data_type=="level":
            trait_data=self.trait_sample_only()
        elif self.data_type=="state_anno_trait_evi_level":
          
            trait_data=self.trait_sample_with_GT_state()

        

        final_data=[]
        if self.data_type!="both":
            for sample_data in trait_data:
                target_speaker,target_dim,dialogue_num,level,dlg_id,input,output=self.get_input_output(sample_data,self.data_type)
                final_data.append({"target_speaker":target_speaker,"target_dim":target_dim,"dialogue_num":dialogue_num,"level":level,"dlg_id":dlg_id,"input":input,"output":output})
        else:
            for sample_data in trait_data:
                target_speaker,target_dim,dialogue_num,level,dlg_id,input,output=self.get_input_output(sample_data,"evi_level")
                final_data.append({"target_speaker":target_speaker,"target_dim":target_dim,"dialogue_num":dialogue_num,"level":level,"dlg_id":dlg_id,"input":input,"output":output})
                target_speaker,target_dim,dialogue_num,level,dlg_id,input,output=self.get_input_output(sample_data,"level")
                final_data.append({"target_speaker":target_speaker,"target_dim":target_dim,"dialogue_num":dialogue_num,"level":level,"dlg_id":dlg_id,"input":input,"output":output})
            #print(input)
            
        return final_data
        

    def get_input_output(self,sample_data,data_type):

        BF_sub_dims={
            '开放性': ["好奇心","审美能力","想象力"],
            '尽责性': ["条理性","效率性","负责性"],
            '外向性': ["社交性","果断性","活力水平"],
            '宜人性': ["同情心","谦恭性","信任度"],
            '神经质性': ["焦虑程度","抑郁程度","情绪稳定程度"]
        }
        dim=sample_data["target_dim"]
        role_name=sample_data["target_speaker"]
        if data_type=="evi_level":
            
            input=f"""你现在是一位大五人格理论专家。
请根据下面{sample_data["target_speaker"]}参与的所有对话，首先综合分析其性格特征，然后从整体上判断在{sample_data["target_dim"]}维度上的程度。

# 输出格式:
在{BF_sub_dims[dim][0]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
在{BF_sub_dims[dim][1]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
在{BF_sub_dims[dim][2]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
综上所述，判断{role_name}的性格特征为...（{dim}高、{dim}低）。

# 输出示例:{example[self.dim_chi2eng[dim]]}

# 提示:
1. 输出示例中的“...”部分请根据对话内容进行填写。
2. 输出示例中括号里面的数字是对话id，你的回答里面也必须采用这个格式。

# 输入:
{sample_data["dialogues_state_label"]}"""
            output=sample_data["nat_lang"]

        elif data_type=="level":

            input=f"""你现在是一位大五人格理论专家。
请根据下面{sample_data["target_speaker"]}，整体上判断在{sample_data["target_dim"]}维度上的程度。
{sample_data["target_dim"]}程度在以下3个选项中选择：无法判断、{sample_data["target_dim"]}高、{sample_data["target_dim"]}低。

# 输入:
{sample_data["dialogues_state_label"]}
"""
            output=sample_data["level"]

            
        elif data_type=="state_anno_trait_evi_level": 
            
            input=f"""你现在是一位大五人格理论专家。
请根据下面{sample_data["target_speaker"]}参与的所有对话和单个对话中体现的性格状态，首先综合分析其性格特征，然后从整体上判断在{sample_data["target_dim"]}维度上的程度。

# 输出格式:
在{BF_sub_dims[dim][0]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
在{BF_sub_dims[dim][1]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
在{BF_sub_dims[dim][2]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
综上所述，判断{role_name}的性格特征为...（{dim}高、{dim}低）。

# 输出示例:{example[self.dim_chi2eng[dim]]}

# 提示:
1. 输出示例中的“...”部分请根据对话内容进行填写。
2. 输出示例中括号里面的数字是对话id，你的回答里面也必须采用这个格式。

# 输入:
{sample_data["dialogues_state_label"]}"""
            output=sample_data["nat_lang"]
        
        elif data_type=="state_anno": 
            
            input=f"""你现在是一位大五人格理论专家。
请根据下面{sample_data["target_speaker"]}参与的所有单个对话中体现的性格状态，首先综合分析其性格特征，然后从整体上判断在{sample_data["target_dim"]}维度上的程度。

# 输出格式:
在{BF_sub_dims[dim][0]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
在{BF_sub_dims[dim][1]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
在{BF_sub_dims[dim][2]}方面，...（{role_name}在对话中的表现）。这体现了{role_name}...(体现的特征)。
综上所述，判断{role_name}的性格特征为...（{dim}高、{dim}低）。

# 输出示例:{example[self.dim_chi2eng[dim]]}

# 提示:
1. 输出示例中的“...”部分请根据对话内容进行填写。
2. 输出示例中括号里面的数字是对话id，你的回答里面也必须采用这个格式。

# 输入:
{sample_data["dialogues_state_label"]}"""
            output=sample_data["nat_lang"]

            
        return sample_data["target_speaker"],sample_data["target_dim"],sample_data["dialogue_num"],sample_data["level"],sample_data["dlg_id"],input,output



if __name__ == '__main__':

    data_formatter=DataFormatter("xxx/dialogue.json",
                                 "xxx/state_annotation.json",
                                 "xxx/trait_annotation.json",
                                 "state_anno",fold_num="fold_1",train_or_test="test")

    

    data=data_formatter.convert2chatglm3()

    print(data[2]["target_dim"])
    print(data[2]["input"])
    print(data[2]["output"])
