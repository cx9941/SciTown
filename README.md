# SciTown

### vllm配置
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -c pytorch

pip install vllm

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source $HOME/.cargo/env


### 运行命令
cd src

python -m test_researcher.main --model_name deepseek-v3

批量运行命令

sh scripts/run_qwen.sh

sh scripts/run_deepseek.sh

### 查看进度命令
ps aux | grep test_researcher.main

可以针对pid进行kill