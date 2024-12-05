import concurrent.futures
import json
import logging
import time
import requests
import os
import threading
import argparse



from data_formatter import DataFormatter

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, help="dataset split")
parser.add_argument('--dia_path', type=str, help="dialogue path")

args = parser.parse_args()

# Global Configuration
MAX_WORKERS = 10
MAX_ARTICLES_PER_RUN = 100000
API_ENDPOINT = ""
MODEL_NAME = "gpt-4-turbo-2024-04-09" # gpt-3.5-turbo   gpt-4-turbo-2024-04-09
API_KEYS = ["",
            "",
            ]

def read_json(read_path):
    assert read_path.split('.')[-1] == 'json'
    with open(read_path, 'r') as f:
        data = json.load(f)
    return data


# Worker function for making the API request
def worker(data: dict):

    left_context = {
        "dia_id":data["dia_id"],
        "target_speaker":data["target_speaker"],
        "target_dim":data["target_dim"],
        "level":data["level"],
        "target_utt_ids":data["target_utt_ids"],
        "output":data["output"]
    }
    
    input=data["input"]
                

    thread_id = threading.current_thread().ident
    api_key = API_KEYS.pop(0)
    API_KEYS.append(api_key)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        'Connection':'close'
    }
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": input}],
        "temperature" : 0.3
    }

    logger.info(f"Thread {thread_id} started with key {api_key}")

    try:
        response = requests.post(url=API_ENDPOINT, headers=headers, data=json.dumps(body), timeout=(180, 180))
        logger.info(f"Thread {thread_id} finished")
        return response.json()["choices"][0]["message"]["content"], left_context
    except Exception as e:
        left_context["input"]=data["input"]
        logger.error(f"{e}, in Thread {thread_id}, Sample: \n{left_context}")
        return None, None

# Main function for executing the workers
def run(data, start, stop):
    success_num, error_num = 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, d) for d in data[start:stop]]

        for future in concurrent.futures.as_completed(futures):
            result, left_context = future.result()

            
            if result:
                save_data={
                "dia_id":left_context["dia_id"],
                "target_speaker":left_context["target_speaker"],
                "target_dim":left_context["target_dim"],
                "level":left_context["level"],
                "target_utt_ids":left_context["target_utt_ids"],
                "label":left_context["output"],
                "pre":result}
                result_file.write(json.dumps(save_data,ensure_ascii=False) + '\n')
                success_num += 1
            else:
                #error_file.write(json.dumps({"sample_idx":idx},ensure_ascii=False) + '\n')
                error_num += 1

    return success_num, error_num

if __name__ == "__main__":
    # Set up directories and file paths
    
    
    current_time_suffix = time.strftime("%Y%m%d%H%M%S", time.localtime())
    result_file_path = f"./results/{MODEL_NAME}_{args.split}_result_{current_time_suffix}.txt"
    error_file_path = f"./results/{MODEL_NAME}_{args.split}_error_sample_{current_time_suffix}.txt"
    log_file_path = f"./logs/{MODEL_NAME}_{args.split}_logs_{current_time_suffix}.log"

   
    # Read and preprocess data
    dialogues=read_json(args.dia_path)
    state_anno_path=f"/data6/sl/EMNLP_pers_evidence/dataset/Chinese/state_annotation/{args.split}.json"
    data_formatter=DataFormatter(args.dia_path,state_anno_path,"ZS_evi_level")
    data_list=data_formatter.convert2chatglm3()
    full_data_dict={f"{data['target_speaker']}_{data['dia_id']}_{data['target_dim']}":data  for data in data_list}
    #data_list=data_list[:1]
    #print(data_list[0]["prompt"])

    # completed_data
    with open(f"./results/final/gpt-4-turbo-2024-04-09.txt", 'r') as data_file:
        completed_data_list = [json.loads(line) for line in data_file]
    completed_data_dict={f"{data['target_speaker']}_{data['dia_id']}_{data['target_dim']}":data  for data in completed_data_list}

    # left_data
    left_data_dict={k:v for k,v in full_data_dict.items() if k not in completed_data_dict.keys()}
    data_list=[v for k,v in left_data_dict.items()]
    #data_list=data_list[:1]
    # Set up logging
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Run the worker function for a subset of data
    start, stop = 0, min(MAX_ARTICLES_PER_RUN, len(data_list))
    logger.info(f"Running for Samples {start + 1} to {stop}")

    with open(result_file_path, "w", encoding = 'utf8') as result_file:
        success_num, error_num = run(data_list, start, stop)
    
    # open(error_file_path, 'w',encoding = 'utf8') as error_file

    logger.info(f"Total sample: {stop - start}, Successful: {success_num}, Errors: {error_num}")
    


