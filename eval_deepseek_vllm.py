import os
import json
import argparse
import time
from src.tot.tasks import *
from src.tot.models_deepseek_vllm import deepseek

# construct a list of dictionary for each problem: {idx: int, input x: str, candidate_0th_steps: list of str, chosen_0th_steps: list of str} --> list of dicts.
def log_info(args):
# open the GPT-4 log file
    file_path = args.log_file_name
    with open (file_path, 'r') as log_file:
        data = json.load(log_file)

    gpt_log_lst = []
    for i, info_dict in enumerate(data):
        if any(d["r"] == 1 for d in info_dict['infos']): # True if any is 1
            gpt_log_lst.append({'idx': i + 900, 
                                'x': info_dict["steps"][0]["x"], 
                                'proposed_0th_steps': info_dict["steps"][0]["new_ys"], 
                                'number of proposals': len(info_dict["steps"][0]["new_ys"]),
                                'selected_0th_steps': info_dict["steps"][0]["select_new_ys"]})
    return gpt_log_lst

# flattens the 2d gpt_log_lst to 1d and turn each proposal string to an input prompt.
# returns: 1d list
def make_eval_prompt(gpt_log_lst) -> tuple[list, dict]:
    eval_prompt_list = []
    for log_dict in gpt_log_lst:
        eval_prompt_list.extend([task.value_prompt_wrap(log_dict['x'], proposal) for proposal in log_dict['proposed_0th_steps']])
    return eval_prompt_list

# return a 2d list: [for each proposal string (which is 1 list): [3*evaluation strings]]
def batch_inference_results(eval_prompt_list, args):
    batch_eval_responses = deepseek(eval_prompt_list, model=args.backend, temperature=args.temperature, max_tokens=args.max_token, n=args.n_evaluate_sample)
    return batch_eval_responses

# create a 3d list: [segment for each probelm from batch_eval_responses]
def inference_results_3d(eval_prompt_list, batch_eval_responses, gpt_log_lst):
    min_idx = 0
    eval_prompt_list_log = []
    eval_responses_list = [] # 3d list
    values_list = [] # 2d list, each inner list contains the values for 1 problem
    for log_dict in gpt_log_lst:
        max_idx = min_idx + log_dict['number of proposals']
        eval_prompt_list_log.append(eval_prompt_list[min_idx:max_idx])
        generate_value_list = [batch_eval_responses[i] for i in range(min_idx, max_idx)] # a list of verbal verdicts for all proposals for the problem
        #     value = task.value_outputs_unwrap(input x, a single gpt4 proposal, [3 verbal verdicts]) 
        values = [task.value_outputs_unwrap(log_dict['x'], proposal, generate_value_list[j]) for j, proposal in enumerate(log_dict['proposed_0th_steps'])]
        eval_responses_list.append(generate_value_list)
        values_list.append(values)
        min_idx = max_idx
    return eval_prompt_list_log, eval_responses_list, values_list

# helper function to check consistency between deepseek and gpt-4, return bools for 2 metrics of consistency checks.
def check_consistency(gpt_selects_lst, deepseek_selects_lst):
    set1, set2 = set(gpt_selects_lst), set(deepseek_selects_lst)
    return len(set1 & set2) / len(set2)

def deepseek_eval_choice(args, gpt_log_lst, task, to_print=True):
    deepseek_logfile = f'./logs/{args.task}/{args.backend}_evaluate_gpt4_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(deepseek_logfile), exist_ok=True)  # exist_ok=True: if the directory already exists, skip creating it with no error raised 
    deepseek_verdicts_log, precision_log = [], []

    eval_prompt_list = make_eval_prompt(gpt_log_lst)
    batch_eval_responses = batch_inference_results(eval_prompt_list, args)
    eval_prompt_list_log, eval_responses_list, values_list = inference_results_3d(eval_prompt_list, batch_eval_responses, gpt_log_lst)

    for i, gpt_dict in enumerate(gpt_log_lst):
        new_ys = gpt_dict['proposed_0th_steps']
        values = values_list[i]
        # selects top solutions
        ids = list(range(len(new_ys)))
        select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        # get the list of deepseek selected 0th steps
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # get consistency values
        precision_score = check_consistency(gpt_dict['selected_0th_steps'], select_new_ys)
        precision_log.append(precision_score)
        
        # log outputs for each problem in a dictionary:
        deepseek_verdicts_log.append({'idx': gpt_dict['idx'], # idx in the dataset
                                      'x': gpt_dict['x'], # input str
                                      'gpt4_0th_step_proposals': gpt_dict['proposed_0th_steps'], # list
                                      'evaluation input prompt': eval_prompt_list_log[i],
                                      'deepseek_evaluations': eval_responses_list[i], # verbal verdicts
                                      'deepseek_verdicts':values, # list of floats
                                      'gpt4_selections': gpt_dict['selected_0th_steps'], # list of strs
                                      'deepseek_selections': select_new_ys, # list of strs
                                      'precision': precision_score, # float
                                      'average precision': sum(precision_log)/len(precision_log) # float
                                       })
        with open(deepseek_logfile, 'w') as f:
            json.dump(deepseek_verdicts_log, f, indent=4)

        if to_print: 
            print(f"-- gpt4_selections --: {gpt_dict['selected_0th_steps']}\n-- deepseek_selections --: {select_new_ys}\n-- precision --: {precision_score}")

    if to_print: 
        print(f"--total average precision--: {deepseek_verdicts_log[-1]['average precision']}")

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['deepseek-ai/DeepSeek-V2-Lite-Chat'], default='deepseek-ai/DeepSeek-V2-Lite-Chat')
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--max_token', type=int, default=512)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--log_file_name', type=str, default='./logs/game24/gpt-4_0.7_propose1_value3_greedy5_start900_end1000.json')
    
    args = args.parse_args()
    return args

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print(args)
    gpt_log_lst = log_info(args)
    task = get_task(args.task)
    deepseek_eval_choice(args, gpt_log_lst, task, to_print=True)
    end_time = time.time()
    elapsed_time = round((end_time - start_time) / 60, 2)
    print(f'total time: {elapsed_time} minutes') 
