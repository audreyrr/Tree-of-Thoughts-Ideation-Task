import argparse
import json
import os
import re
from pathlib import Path

from tqdm import tqdm

from tot.olmo2_7b_instruct_vllm import olmo2_7b_instruct


def load_math100(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Some HF exports may be {"data": [...]} or similar.
        for key in ["data", "test", "train"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"Unsupported JSON dict format. Keys: {list(data.keys())}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples, got {type(data)}")

    return data


def get_field(example, names, default=None):
    for name in names:
        if name in example and example[name] is not None:
            return example[name]
    return default


# normalize answer string for comparison:
#   \\boxed{14/3}
#   $\\frac{14}{3}$
#   14/3
#   14
def normalize_answer(ans):

    if ans is None:
        return ""

    ans = str(ans).strip()

    # remove common wrappers
    ans = ans.replace("$", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace(" ", "")

    # convert \frac{a}{b} -> a/b
    ans = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", ans)

    # remove \boxed{...}
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", ans)
    if boxed:
        ans = boxed[-1]

    # strip punctuation
    ans = ans.strip(".。,:;")

    return ans


def extract_boxed_answer(text):
    """
    Extract the final answer from model output.
    Prefer:
      \\boxed{...}
      Answer: ...
      final answer is ...
    Otherwise fall back to the last non-empty line.
    """
    if not text:
        return ""

    # prefer boxed answer
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Answer: ...
    patterns = [
        r"(?i)answer\s*:\s*(.+)",
        r"(?i)final answer\s*(?:is|:)?\s*(.+)",
        r"(?i)therefore[, ]+(.+)",
    ]

    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            candidate = matches[-1].strip()
            candidate = candidate.split("\n")[0].strip()
            return candidate

    # fall back to last non-empty line
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        return lines[-1]

    return text.strip()


def make_cot_prompt(problem):
    return f"""Solve the following math problem step by step. Put your final answer in \\boxed{{}}.

Problem:
{problem}

Solution:
"""


def evaluate_prediction(pred_text, gold_answer):
    pred_answer = extract_boxed_answer(pred_text)

    norm_pred = normalize_answer(pred_answer)
    norm_gold = normalize_answer(gold_answer)

    exact = int(norm_pred == norm_gold and norm_gold != "")

    return {
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "normalized_pred_answer": norm_pred,
        "normalized_gold_answer": norm_gold,
        "correct": exact,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/projectnb/tin-lab/audrey/tree-of-thoughts/src/tot/data/MATH100/test.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/projectnb/tin-lab/audrey/tree-of-thoughts/logs/math100/olmo2_7b_instruct_vllm_cot.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMo-2-1124-7B-Instruct",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    data = load_math100(args.data_path)

    end = args.end if args.end is not None else len(data)
    examples = data[args.start:end]

    print(f"Loaded {len(data)} total examples from {args.data_path}")
    print(f"Running examples [{args.start}, {end}) = {len(examples)} examples")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    logs = []
    total_correct = 0
    total_count = 0

    for batch_start in tqdm(range(0, len(examples), args.batch_size)):
        batch = examples[batch_start : batch_start + args.batch_size]

        chat_prompts = []
        metadata = []

        for local_i, ex in enumerate(batch):
            global_idx = args.start + batch_start + local_i

            problem = get_field(ex, ["problem", "question", "input", "prompt"])
            gold_answer = get_field(ex, ["answer", "final_answer", "target", "output"])

            if problem is None:
                raise ValueError(f"Could not find problem field in example {global_idx}: {ex.keys()}")

            if gold_answer is None:
                # if datasets only have solution, then try extracting boxed answer from solution.
                solution = get_field(ex, ["solution", "rationale"], "")
                gold_answer = extract_boxed_answer(solution)

            prompt = make_cot_prompt(problem)

            chat_prompts.append(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )

            metadata.append(
                {
                    "idx": global_idx,
                    "problem": problem,
                    "gold_answer": gold_answer,
                    "raw_example": ex,
                    "prompt": prompt,
                }
            )

        # olmo2_7b_instruct returns a 2D list:
        # responses[prompt_idx][sample_idx]
        responses = olmo2_7b_instruct(
            chat_prompts,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.n,
            seed_value=args.seed,
        )

        for meta, response_list in zip(metadata, responses):
            sample_results = []

            for sample_id, pred_text in enumerate(response_list):
                eval_info = evaluate_prediction(pred_text, meta["gold_answer"])

                sample_results.append(
                    {
                        "sample_id": sample_id,
                        "output": pred_text,
                        **eval_info,
                    }
                )

            # For n > 1, use "any correct" as the per-problem success.
            any_correct = int(any(r["correct"] for r in sample_results))

            total_correct += any_correct
            total_count += 1

            logs.append(
                {
                    "idx": meta["idx"],
                    "problem": meta["problem"],
                    "gold_answer": meta["gold_answer"],
                    "prompt": meta["prompt"],
                    "samples": sample_results,
                    "any_correct": any_correct,
                    "running_accuracy": total_correct / total_count,
                }
            )

        # Save after every batch so partial results are preserved.
        with open(args.output_path, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "accuracy": total_correct / total_count if total_count else 0.0,
                    "num_correct": total_correct,
                    "num_total": total_count,
                    "logs": logs,
                },
                f,
                indent=2,
            )

        print(
            f"Saved {total_count} examples. "
            f"Accuracy so far: {total_correct}/{total_count} = {total_correct / total_count:.4f}"
        )

    print("Done.")
    print(f"Final accuracy: {total_correct}/{total_count} = {total_correct / total_count:.4f}")
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
