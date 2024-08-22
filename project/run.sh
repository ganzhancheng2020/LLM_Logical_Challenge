#!/bin/bash

# 检查 Python 命令是否存在
if ! command -v python &> /dev/null; then
    echo "Python 未安装或不在PATH中，请先安装 Python 并确保它可以被正确调用。"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "pip 未安装或不在PATH中，请先安装 pip。"
    exit 1
fi

# 安装依赖项
echo "正在安装依赖项..."
pip install -r requirements.txt

# 检查依赖项安装是否成功
if [ $? -ne 0 ]; then
    echo "依赖项安装失败，退出脚本。"
    exit 1
fi

# 定义 src 目录的路径
src_dir="./src"

# 检查 src 目录是否存在
if [ ! -d "$src_dir" ]; then
    echo "src 目录不存在，请确认 src 目录存在并且包含所需的 Python 文件。"
    exit 1
fi

# 执行 train_llama3_lora.py
echo "正在运行 train_llama3_lora.py..."
python "${src_dir}/train_llama3_lora.py"

# 检查 train_llama3_lora.py 是否成功执行
if [ $? -ne 0 ]; then
    echo "train_llama3_lora.py 运行失败，退出脚本。"
    exit 1
fi

# 启动 vllm API 服务器
echo "正在启动 vllm API 服务器..."
python -m vllm.entrypoints.openai.api_server \
    --model /tmp/merged_model_an \
    --served-model-name llama3-8b-lora \
    --max-model-len=4096 \
    --trust-remote-code

# 执行 run.py
echo "正在运行 run.py..."
python "${src_dir}/run.py"

# 检查 run.py 是否成功执行
if [ $? -ne 0 ]; then
    echo "run.py 运行失败，退出脚本。"
    exit 1
fi

# 执行 runcheck.py
echo "正在运行 runcheck.py..."
python "${src_dir}/runcheck.py"

# 检查 runcheck.py 是否成功执行
if [ $? -ne 0 ]; then
    echo "runcheck.py 运行失败，退出脚本。"
    exit 1
fi

echo "所有脚本运行完成。"