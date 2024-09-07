
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from llama_index.core import SimpleDirectoryReader
from ReadLoad import read_jsonl, write_jsonl
import json
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# In[2]:


def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

ana = read_json('data/external_data/ana.json')
qwen_data = read_json('data/external_data/qwen_train_data_1697.json')


# In[5]:


#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('xorbits/bge-small-zh-v1.5', cache_dir="./model")


# In[6]:


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 假设 model_dir 是你下载模型的本地路径
model_path = model_dir

# 使用本地模型路径创建 HuggingFaceEmbedding 实例
embedding = HuggingFaceEmbedding(
    model_name=model_path,  # 使用本地模型路径
    cache_folder=model_dir,  # 指定缓存目录
    embed_batch_size=128
)

Settings.embed_model = embedding


# In[9]:
rag_data = ana + qwen_data

qa_dict = {}
questions = []
for i, data in enumerate(rag_data):
    question = data['instruction'].split("题目如下：")[1]
    answer = data['output']
    id_ = i
    data['answer'] = answer
    data['question'] = question
    data['rag_id'] = id_
    qa_dict[id_] = answer


# In[16]:


documents = [Document(text=t['question'], metadata={"rag_id": t['rag_id']}) for t in rag_data]
vector_index = VectorStoreIndex.from_documents(documents)

import torch
torch.cuda.empty_cache()
# In[31]:


def get_few_shot(query, k=3):
    vector_retriever = vector_index.as_retriever(similarity_top_k=k)
    nodes = vector_retriever.retrieve(query)
    string = ''
    for node in nodes:
        question = node.text
        answer_id = node.metadata['rag_id']
        answer = qa_dict[answer_id]
        qa = '\n### 回答:\n'.join([question,answer])
        string += qa + '\n\n'
    return string

