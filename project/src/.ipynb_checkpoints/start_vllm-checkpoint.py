
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('python -m vllm.entrypoints.openai.api_server --model /tmp/merged_model_an  --served-model-name llama3-8b-lora --max-model-len=4096 --trust-remote-code')

