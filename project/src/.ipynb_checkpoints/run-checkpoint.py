import json
import os
from pprint import pprint
import re
from tqdm import tqdm
import random

import uuid
import openai
from openai import OpenAI
import tiktoken
import json
import numpy as np
import requests
from scipy import sparse
#from rank_bm25 import BM25Okapi
#import jieba
from http import HTTPStatus


from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import json
import time
from tqdm import tqdm

logger.remove()  # 移除默认的控制台输出
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="INFO", rotation="00:00", retention="10 days", compression="zip")

MODEL_NAME = 'llama3-7B-Instruct-lora'
def api_retry(MODEL_NAME, query):
    max_retries = 5
    retry_delay = 60  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            return call_qwen_api(MODEL_NAME, query)
        except Exception as e:
            attempts += 1   
            if attempts < max_retries:
                logger.warning(f"Attempt {attempts} failed for text: {query}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed for text: {query}. Error: {e}")
                raise
def call_qwen_api(MODEL_NAME, query):
    # 这里采用dashscope的api调用模型推理，通过http传输的json封装返回结果

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
    )
    completion = client.chat.completions.create(
      model=MODEL_NAME,
      messages=[
                # {'role':'system','content':'你是一个解决推理任务的专家，你需要分析出问题中的每个实体以及响应关系。然后根据问题一步步推理出结果。并且给出正确的结论。'},

        {"role": "user", "content": query}
      ]
    )
    return completion.choices[0].message.content

# 这里定义了prompt推理模版

def get_prompt(problem, question, options):

    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。题目如下：

### 题目:
{problem}

### 问题:
{question}
{options}
"""
    # print(prompt)
    return prompt


# 这里使用extract抽取模获得抽取的结果

def extract(input_text):
    # 正则表达式匹配“答案是：”后面的第一个字符
    print("input_text is", input_text)
    ans_pattern = re.compile(r"答案是：([A-Za-z0-9])")

    match = ans_pattern.search(input_text)
    print("match.group(1)", match.group(1))
    if not match:
        return 'Error'

    return match.group(1)


def most_frequent_char(char1, char2, char3):
    # 创建一个字典来存储每个字符的出现次数
    frequency = {char1: 0, char2: 0, char3: 0}
    
    # 增加每个字符的出现次数
    frequency[char1] += 1
    frequency[char2] += 1
    frequency[char3] += 1
    
    # 找到出现次数最多的字符
    most_frequent = max(frequency, key=frequency.get)
    
    return most_frequent

'''
def process_datas(datas,MODEL_NAME):
    results = []

    # 送入多线程任务
    for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
        problem = data['problem']
        for id,question in enumerate(data['questions']):
            prompt = get_prompt(problem, 
                                question['question'], 
                                question['options'],
                                    )
            #res,res1,res2 = api_retry(MODEL_NAME, prompt),api_retry(MODEL_NAME, prompt),api_retry(MODEL_NAME, prompt)
            res= api_retry(MODEL_NAME, prompt)
            #extract_response,extract_response1,extract_response2 = extract(res),extract(res1),extract(res2)
            extract_response= extract(res)
            #ans = most_frequent_char(extract_response,extract_response1,extract_response2)
            ans = extract_response
            data['questions'][id]['answer'] = ans
            results.append(data) 
    return results
'''

def process_datas(datas,MODEL_NAME):
    results = []
    # 定义线程池 选择16线程
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 这里我们使用future_data 存储每个线程的数据
        future_data = {}
        # 这里的lens记录了调用api的次数，也就是我们每个问题背景下的所有子问题之和。
        lens = 0
        # 送入多线程任务
        # 这里每个data下是一个问题背景，其中包含多个子问题。
        for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
            problem = data['problem']
            # 这里面我们用enumerate方法每次循环得到问题的序号id和实际的问题。
            for id,question in enumerate(data['questions']):
                prompt = get_prompt(problem, 
                                    question['question'], 
                                    question['options'],
                                    )
                # 这里送入线程池等待处理，使用api_retry，向api_retry传入MODEL_NAME, prompt参数
                future = executor.submit(api_retry, MODEL_NAME, prompt)
                # 每个线程我们存储对应的json问题数据以及问题序号id，这样我们就能定位出执行的是哪个子问题
                future_data[future] = (data,id)
                time.sleep(0.6)  # 控制每0.6秒提交一个任务 防止接口超过并发数
                lens += 1
        # 处理多线程任务
        for future in tqdm(as_completed(future_data), total=lens, desc="Processing tasks"):
            # print('data',data)
            # 取出每个线程中的字典数据及对应的问题id
            data = future_data[future][0]
            problem_id = future_data[future][1]
            try:
                # 获取api运行结果
                res  = future.result()
                # 抽取大语言模型返回结果
                extract_response = extract(res)
                # print('res',extract_response)
                # 装入answer字段
                data['questions'][problem_id]['answer'] = extract_response
                # 在结果列表中新增数据字典
                results.append(data)
                # print('data',data)
                
            except Exception as e:
                logger.error(f"Failed to process text: {data}. Error: {e}")
    
    return results

def has_complete_answer(questions):
    # 这里假设完整答案的判断逻辑是：每个question都有一个'answer'键
    for question in questions:
        if 'answer' not in question:
            return False
    return True

def filter_problems(data):
    result = []
    problem_set = set()

    for item in data:
        # print('处理的item' ,item)
        problem = item['problem']
        if problem in problem_set:
            # 找到已存在的字典
            for existing_item in result:
                if existing_item['problem'] == problem:
                    # 如果当前字典有完整答案，替换已存在的字典
                    if has_complete_answer(item['questions']):
                        existing_item['questions'] = item['questions']
                        existing_item['id'] = item['id']
                    break
        else:
            # 如果当前字典有完整答案，添加到结果列表
            if has_complete_answer(item['questions']):
                result.append(item)
                problem_set.add(problem)

    return result

def find_missing_ids(dict_list):
    # 提取所有序号
    extracted_ids = {int(d['id'][-3:]) for d in dict_list}
    
    # 创建0-500的序号集合
    all_ids = set(range(500))
    
    # 找出缺失的序号
    missing_ids = all_ids - extracted_ids
    
    return sorted(missing_ids)


def main(ifn, ofn):
    if os.path.exists(ofn):
        pass
    data = []
    # 按行读取数据
    with open(ifn) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    datas = data
    # print(data)
    # 均匀地分成多个数据集
    return_list = process_datas(datas,MODEL_NAME)
    print(len(return_list))
    print("All tasks finished!")


    return_list1 = filter_problems(return_list)
    sorted_data = sorted(return_list1, key=lambda x: int(str(x['id'])[-3:]))
    print(sorted_data)

    # 示例字典列表
    dict_list = sorted_data

    # 找出缺失的序号
    missing_ids = find_missing_ids(dict_list)
    print("缺失的序号:", missing_ids)

    len(missing_ids)
    data  = []
    with open(ifn) as reader:
        for id,line in enumerate(reader):
            if(id in missing_ids):
                sample = json.loads(line)
                for question in sample['questions']:
                    question['answer'] = 'A'
                sorted_data.append(sample)
    sorted_data = sorted(sorted_data, key=lambda x: int(str(x['id'])[-3:]))
    with open(ofn, 'w') as writer:
        for sample in sorted_data:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write('\n')

if __name__ == '__main__':
    return_list = main('round1_test_data.jsonl', 'upload.jsonl')
    
