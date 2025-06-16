from vllm import LLM, SamplingParams

# return a 2d list
def deepseek(eval_prompt_list, model='/stage/hf_cache/DeepSeek-V2-Lite-Chat', temperature=0.7, max_tokens=512, n=1, seed_value=1):
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=temperature,max_tokens=max_tokens, n=n, seed=seed_value)
    # Create an LLM. Load model weights to GPU.
    llm = LLM(model=model, trust_remote_code=True, max_model_len=2048)
    outputs = llm.chat(eval_prompt_list, sampling_params)
    responses = []
    for output in outputs:
        responses.append([output.outputs[i].text for i in range(len(output.outputs))])
    return responses  


