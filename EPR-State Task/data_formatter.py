import json
from BigFive_definition import definition
from  answer_example import example

def read_json(read_path):
    assert read_path.split('.')[-1] == 'json'
    with open(read_path, 'r') as f:
        data = json.load(f)
    return data


class DataFormatter:
    def __init__(self,dlg_path,state_anno_path,data_type,train_ratio=1):
        
        # param data_type: ["level", "utt_id", "analysis"]
        #  level: 只用label
        #  utt_id: 加上 evidence utterance id
        #  analysis: 加上自然语言解释

        assert data_type in ["evi_level", "level","both"]
        self.data_type=data_type
        

        self.dialogue=read_json(dlg_path)
        self.state_anno=read_json(state_anno_path)

        self.train_ratio=train_ratio

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
        
        self.state_sample()
       
        

    
    def state_sample(self):
        state_data=[]
        for role_name,role_data in self.state_anno.items():
            for dia_id,all_dim_data in role_data["annotation"].items():
                sample_dialogue=self.dialogue[role_name]["dialogue"][dia_id]
                target_utt_ids=[]
                for id,utt in enumerate(sample_dialogue):
                    if role_name in utt.split('：')[0]:
                        target_utt_ids.append(id+1)
                for dim_name,dim_data in all_dim_data.items():
                    sample_data={
                        "dia_id":dia_id,
                        "target_speaker":role_name,
                        "target_dim":self.dim_eng2chi[dim_name],
                        "dialogue":sample_dialogue,
                        "target_utt_ids":target_utt_ids,
                        "level":dim_data["level"],
                        "utt_id":dim_data["utt_id"],
                        "nat_lang":dim_data["nat_lang"]

                    }
                    state_data.append(sample_data)
        return  state_data

 

    def convert2chatglm3(self):

        state_data=self.state_sample()
        final_data=[] 
        if self.data_type!="both":
            for sample_data in state_data:
                dia_id,target_speaker,target_dim,level,target_utt_ids,input,output=self.get_input_output(sample_data,self.data_type)
                final_data.append({"dia_id":dia_id,"target_speaker":target_speaker,"target_dim":target_dim,"level":level,"target_utt_ids":target_utt_ids,"input":input,"output":output})
        else:
            for sample_data in state_data:
                dia_id,target_speaker,target_dim,level,target_utt_ids,input,output=self.get_input_output(sample_data,"evi_level")
                final_data.append({"dia_id":dia_id,"target_speaker":target_speaker,"target_dim":target_dim,"level":level,"target_utt_ids":target_utt_ids,"input":input,"output":output})

                dia_id,target_speaker,target_dim,level,target_utt_ids,input,output=self.get_input_output(sample_data,"level")
                final_data.append({"dia_id":dia_id,"target_speaker":target_speaker,"target_dim":target_dim,"level":level,"target_utt_ids":target_utt_ids,"input":input,"output":output})
        
        final_data=final_data[:int(len(final_data)*self.train_ratio)]

        return final_data

    def convert2baichuan2(self):
        state_data=self.state_sample()
        final_data=[]
        if self.data_type!="both":
            for sample_data in state_data:
                dia_id,target_speaker,target_dim,level,target_utt_ids,input,output=self.get_input_output(sample_data,self.data_type)
                conversation=[
                    {"from": "human","value": input},
                    {"from": "gpt","value": output}]
            
                final_data.append({"dia_id":dia_id,"target_speaker":target_speaker,"target_dim":target_dim,"level":level,"target_utt_ids":target_utt_ids,"conversations":conversation})
        else:
            for sample_data in state_data:
                dia_id,target_speaker,target_dim,level,target_utt_ids,input,output=self.get_input_output(sample_data,"evi_level")
                conversation=[
                    {"from": "human","value": input},
                    {"from": "gpt","value": output}]
                final_data.append({"dia_id":dia_id,"target_speaker":target_speaker,"target_dim":target_dim,"level":level,"target_utt_ids":target_utt_ids,"conversations":conversation})

                dia_id,target_speaker,target_dim,level,target_utt_ids,input,output=self.get_input_output(sample_data,"level")
                conversation=[
                    {"from": "human","value": input},
                    {"from": "gpt","value": output}]
                final_data.append({"dia_id":dia_id,"target_speaker":target_speaker,"target_dim":target_dim,"level":level,"target_utt_ids":target_utt_ids,"conversations":conversation})
        return final_data
        

    def get_input_output(self,sample_data,data_type):

        if data_type=="evi_level":
            dia_string="对话："
            for utterance in sample_data["dialogue"]:
                dia_string+='\n'+utterance

            input=f"""你现在是一位大五人格理论专家。
请根据以下对话内容，首先分析对话中体现{sample_data["target_speaker"]}{sample_data["target_dim"]}程度的句子id，然后概括这些句子的内容并判断{sample_data["target_speaker"]}的{sample_data["target_dim"]}程度。
{sample_data["target_dim"]}程度在以下3个选项中选择：无法判断、{sample_data["target_dim"]}高、{sample_data["target_dim"]}低。

{dia_string}"""

            if sample_data["level"]=="无法判断":
                output=f'在这段对话中，没有体现{sample_data["target_speaker"]}{sample_data["target_dim"]}特征的句子。因此，{sample_data["nat_lang"].split("因此，")[-1]}'
            else:
                output=f'根据对话中第{sample_data["utt_id"]}句，{sample_data["nat_lang"].replace("在这段对话中，","")}'


        elif data_type=="level":

            dia_string="对话："
            for utterance in sample_data["dialogue"]:
                dia_string+='\n'+utterance

            input=f"""你现在是一位大五人格理论专家。
请根据以下对话内容，判断{sample_data["target_speaker"]}的{sample_data["target_dim"]}程度。
{sample_data["target_dim"]}程度在以下3个选项中选择：无法判断、{sample_data["target_dim"]}高、{sample_data["target_dim"]}低。

{dia_string}"""
            
            output=sample_data["level"]

            
        return sample_data["dia_id"],sample_data["target_speaker"],sample_data["target_dim"],sample_data["level"],sample_data["target_utt_ids"],input,output



if __name__ == '__main__':

    data_formatter=DataFormatter("xxx/dialogue.json",
                                 "xxx/train_annotation.json",
                                 "ZS_evi_level")

    

    #data=data_formatter.convert2baichuan2()
    data=data_formatter.convert2chatglm3()

    print(data[1110]["target_dim"])
    print(data[1110]["input"])
    print(data[1110]["output"])
