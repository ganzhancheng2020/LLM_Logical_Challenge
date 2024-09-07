### 运行环境
硬件要求：至少配备有 24GB 显存的 GPU。
软件依赖：请参考 requirements.txt 文件安装所有必要的 Python 包。

### 启动说明
确保所有依赖已通过 requirements.txt 文件安装。
运行脚本 run.sh 来依次执行各个步骤。

### 代码说明
train_llama3_lora.py：用于训练模型。
run.py：执行模型推理，读取 round1_test_data.jsonl 文件，并生成初步结果 upload.jsonl。
runcheck.py：检查并完善 upload.jsonl 文件，最终输出可提交的submit.jsonl文件。