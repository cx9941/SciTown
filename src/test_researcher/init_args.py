import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='qwen', type=str)
parser.add_argument("--task_name", default='crossdisc', type=str)
parser.add_argument("--query", default='深海探测研究', type=str)
args = parser.parse_args()
args.log_dir = f"../outputs/log/{args.task_name}/{args.model_name}"
args.execution_logs_dir = f"../outputs/execution_log/{args.task_name}/{args.model_name}"
os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.execution_logs_dir, exist_ok=True)