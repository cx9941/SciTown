import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='qwen', type=str)
parser.add_argument("--task_name", default='crossdisc', type=str)
parser.add_argument("--query", default='复杂海洋环境下基于多模态深度学习的智能探测与认知研究', type=str)
args = parser.parse_args()
args.log_dir = f"../outputs/log/{args.task_name}/{args.query}/{args.model_name}"
args.execution_logs_dir = f"../outputs/execution_log/{args.task_name}/{args.query}/{args.model_name}"
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.execution_logs_dir, exist_ok=True)

# 获取当前时间戳，格式如 20250415_142530
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

args.output_log_file = f"{args.log_dir}/{timestamp}.json"
args.task_execution_output_json_path = f"{args.execution_logs_dir}/{timestamp}.json"