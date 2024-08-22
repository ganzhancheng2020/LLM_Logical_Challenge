
#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# In[ ]:


import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('./LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir='./models/Meta-Llama-3.1-8B-Instruct', revision='master')


# In[2]:


# model_dir = "./models/LLM-Research/Meta-Llama-3___1-8B-Instruct"


# ## 这里运行完请重启notebook

# In[2]:


from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig


# In[3]:


import pandas as pd
# 将JSON文件转换为CSV文件
train_data_size = 4000
ana = pd.read_json('data/external_data/ana.json')
qwen = pd.read_json('data/external_data/qwen_train_data_1697.json')
gsm8k2 = pd.read_json('data/external_data/gsm8k2_train_data_7473.json')
#gsm8k2.iloc[:num_records_needed]
temp_df = pd.concat([ana, qwen,gsm8k2], ignore_index=True)
combined_df = temp_df.iloc[:train_data_size]
combined_df = combined_df.sample(frac=1)
ds = Dataset.from_pandas(combined_df.iloc[:int(train_data_size*0.8)])
dev_ds = Dataset.from_pandas(combined_df.iloc[int(train_data_size*0.8):])


# In[4]:


print(len(ds),len(dev_ds))


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer


# In[6]:


def process_func(example):
    MAX_LENGTH = 1800    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是一个逻辑推理专家，擅长解决逻辑推理问题。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# In[7]:


tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id
dev_id = dev_ds.map(process_func, remove_columns=ds.column_names)
dev_id


# In[8]:


tokenizer.decode(tokenized_id[0]['input_ids'])


# In[9]:


tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))


# In[10]:


import torch

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",torch_dtype=torch.bfloat16)
model


# In[11]:


model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法


# In[12]:


model.dtype


# In[13]:


from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config


# In[14]:


model = get_peft_model(model, config)
config


# In[15]:


model.print_trainable_parameters()


# In[ ]:


# args = TrainingArguments(
#     output_dir="./output/Qwen2_instruct_lora",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     logging_steps=10,
#     num_train_epochs=1,
#     save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
#     learning_rate=1e-4,
#     save_on_each_node=True,
#     gradient_checkpointing=True
# )
output_dir="./output/llama3_instruct_lora/20240818"
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    logging_steps=100,
    eval_steps=100,
    num_train_epochs=5,
    save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
    learning_rate=1e-4,
    save_on_each_node=True,
    #bf16=True,
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    evaluation_strategy="steps"
    # save_strategy="steps"
)


# In[17]:


# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_id,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
# )
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    eval_dataset=dev_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()


# In[ ]:


torch.backends.cuda.enable_mem_efficient_sdp(False)


# ## 这里运行完请重启notebook

# In[1]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

output_dir="./output/llama3_instruct_lora/20240818"

mode_path = "./models/Meta-Llama-3.1-8B-Instruct/LLM-Research"
lora_path = f'{output_dir}/checkpoint-700' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.float16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = '''你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。
题目如下：\n\n### 题目:\n假设您需要构建一个二叉搜索树，其中每个节点或者是一个空的节点（称为"空节点"），或者是一个包含一个整数值和两个子树的节点（称为"数值节点"）。以下是构建这棵树的规则：\n\n1. 树中不存在重复的元素。\n2. 对于每个数值节点，其左子树的所有值都小于该节点的值，其右子树的所有值都大于该节点的值。\n3. 插入一个新值到一个"空节点"时，该"空节点"会被一个包含新值的新的数值节点取代。\n4. 插入一个已存在的数值将不会改变树。\n\n请基于以上规则，回答以下选择题：
\n\n### 问题:\n选择题 1：\n给定一个空的二叉搜索树，插入下列数字: [5, 9, 2, 10, 11, 3]，下面哪个选项正确描述了结果树的结构？\nA. tree(5, tree(2, tree(3, nil, nil), nil), tree(9, tree(10, nil, nil), tree(11, nil, nil)))\nB. tree(5, tree(2, nil, tree(3, nil, nil)), tree(9, nil, tree(10, nil, tree(11, nil, nil))))\nC. tree(5, tree(3, tree(2, nil, nil), nil), tree(9, nil, tree(10, tree(11, nil, nil), nil)))\nD. tree(5, nil, tree(2, nil, tree(3, nil, nil)), tree(9, tree(11, nil, nil), tree(10, nil, nil)))'''
inputs = tokenizer.apply_chat_template([{"role": "system", "content": "你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为\"答案是：A。"},{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to('cuda')
#tensor_gpu = tensor.to('cuda:0')

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# In[2]:


# 模型合并存储

new_model_directory = "/tmp/merged_model_an"
merged_model = model.merge_and_unload()
# 将权重保存为safetensors格式的权重, 且每个权重文件最大不超过2GB(2048MB)
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
# 将tokenizer也保存到 merge_model_dir
tokenizer.save_pretrained(new_model_directory)



# ## 这里运行完请重启notebook
