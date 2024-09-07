import subprocess
import torch
torch.cuda.empty_cache()

# 定义要执行的命令
command = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "./model/Meta-Llama-3.1-8B-Instruct-lora",
    "--served-model-name", "Meta-Llama-3.1-8B-Instruct-lora",
    "--max-model-len=4096"
]

# 启动子进程
process = subprocess.Popen(command)

# # 等待子进程结束
# process.wait()

# 检查进程退出状态
# if process.returncode == 0:
#     print("命令执行成功")
# else:
#     print(f"命令执行失败，退出码：{process.returncode}")