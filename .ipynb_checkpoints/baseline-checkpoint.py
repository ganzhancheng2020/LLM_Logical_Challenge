from multiprocessing import Process, Manager
import json
import os
from pprint import pprint
import re
from tqdm import tqdm
import random

import uuid
import openai
import tiktoken
import json
import numpy as np
import requests
from retry import retry
from scipy import sparse
#from rank_bm25 import BM25Okapi
#import jieba
from http import HTTPStatus
import dashscope

MODEL_NAME = 'qwen1.5-32b-chat'

@retry(delay=3, tries=3)
def call_qwen_api(MODEL_NAME, query):
    messages = [
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(
        MODEL_NAME,
        messages=messages,
        result_format='message',  # set the result is message format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
        return response['output']['choices'][0]['message']['content']
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        raise Exception()



def get_prompt(problem, question, options):

    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。题目如下：

### 题目:
{problem}

### 问题:
{question}
{options}
"""

    return prompt


def extract(input_text):
    ans_pattern = re.compile(r"答案是：(.)", re.S)

    problems = ans_pattern.findall(input_text)
    return problems[0]


def produce(data, MODEL_NAME, return_list, pid):
    tqdm1 = tqdm
    for task in tqdm1(data):
            problem = task['problem']
            for question in task['questions']:

                prompt = get_prompt(problem, 
                                    question['question'], 
                                    question['options'],
                                    )

                response = call_qwen_api(MODEL_NAME, prompt)
                try:
                    extract_response = extract(response)
                    question[MODEL_NAME] = extract_response
                    if pid == 0:
                        pprint(extract_response)
                    break
                except:
                    pass
            
            return_list.append(task)

def main(ifn, ofn):
    if os.path.exists(ofn):
        pass

    POOL_SIZE = 5
    data = []
    with open(ifn) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)

    random.shuffle(data)

    datas = [data[i::POOL_SIZE] for i in range(POOL_SIZE)]

    with Manager() as manager:
        producers = []
        return_list = manager.list()
        for i in range(POOL_SIZE):
            p = Process(target=produce,
                    args=(datas[i],
                        MODEL_NAME,
                        return_list,
                        i,
                        )
                    )
            producers.append(p)

        for p in producers:
            p.start()

        for p in producers:
            p.join()

        print(len(return_list))

        with open(ofn, 'w') as writer:
            for sample in return_list:
                writer.write(json.dumps(sample, ensure_ascii=False))
                writer.write('\n')
        
    print("All tasks finished!")
    evaluate(ofn)


def evaluate(ofn):
    data = []
    with open(ofn) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)

    pse = 0
    cnt = 0
    tot = 0
    for task in data:
        for question in task['questions']:
            
            if MODEL_NAME in question:
                tot += 1
                cnt += question[MODEL_NAME] == question['answer']
            else:
                pse += 1

    print(cnt, tot, cnt/tot, pse)

    

if __name__ == '__main__':

    a = extract("""根据欧几里得算法，逐步解析计算两个数6和7的最大公约数（gcd）的步骤如下：

1. 判断6和7是否相等：不相等。
2. 判断6和7大小关系，7 > 6，所以用更大的数7减去较小的数6得到结果1。
3. 现在计算6和1的最大公约数。
4. 6 > 1，根据算法用更大的数6减去较小的数1得到结果5。
5. 再计算5和1的最大公约数。
6. 5 > 1，用5减去1得到结果4。
7. 再计算4和1的最大公约数。
8. 4 > 1，用4减去1得到结果3。
9. 再计算3和1的最大公约数。
10. 3 > 1，用3减去1得到结果2。
11. 再计算2和1的最大公约数。
12. 2 > 1，用2减去1得到结果1。
13. 最后计算1和1的最大公约数，两数相等，gcd即为这两个数，也就是1。

因此，6和7的最大公约数是1。

答案是：C.""")

    print(a)
    main('round1_train_data.jsonl', 'qwen.jsonl')
