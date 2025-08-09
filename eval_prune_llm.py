import torch
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
from eval import eval_ppl
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks
import json
import lm_eval
import logging


# Import get_loaders function from data module within the same directory
device = torch.device("cuda:0")
model = torch.load(f"save_model/llama2-13b-50%/pruned_model.pt", map_location="cuda", weights_only=False)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", use_fast=False)
model.eval()
model.seqlen = 2048
ppl = eval_ppl(model, tokenizer, device)    
print(f"ppl on wikitext {ppl}")
initialize_tasks()
hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)
task_names = lm_eval_utils.pattern_match(["piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge"], ALL_TASKS)

logging.info(f"Selected Tasks: {task_names}")

results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=0, batch_size=16)['results']

metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
logging.info(json.dumps(metric_vals, indent=4))

def calculate_avg_accuracy(task_names, results):
    n_tasks = len(task_names)
    acc_cumul = sum(result.get('acc_norm,none', result['acc,none']) for task, result in results.items())
    return acc_cumul / n_tasks

acc_avg = calculate_avg_accuracy(task_names, results)
logging.info(f"Average accuracy across tasks: {acc_avg}")
