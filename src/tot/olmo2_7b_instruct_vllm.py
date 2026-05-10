from vllm import LLM, SamplingParams


def olmo2_7b_instruct(
    eval_prompt_list,
    model="allenai/OLMo-2-1124-7B-Instruct",
    temperature=0.7,
    max_tokens=512,
    n=1,
    seed_value=1,
):
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        seed=seed_value,
    )

    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=2048,
    )

    outputs = llm.chat(eval_prompt_list, sampling_params)

    responses = []
    for output in outputs:
        responses.append([candidate.text for candidate in output.outputs])

    return responses


# [
#   [response_1_for_prompt_1, response_2_for_prompt_1, ...],
#   [response_1_for_prompt_2, response_2_for_prompt_2, ...],
#   ...
# ]