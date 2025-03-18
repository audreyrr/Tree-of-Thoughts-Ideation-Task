from vllm import LLM, SamplingParams
import os
cache_dir = "/projectnb2/tin-lab/cache/huggingface/hub"
os.environ["HF_HOME"] = cache_dir

# return a 2d list
def deepseek(eval_prompt_list, model='deepseek-ai/DeepSeek-V2-Lite-Chat', temperature=0.7, max_tokens=512, n=1):
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=temperature,max_tokens=max_tokens, n=n, seed=8)
    # Create an LLM. Load model weights to GPU.
    llm = LLM(model=model, trust_remote_code=True)
    outputs = llm.generate(eval_prompt_list, sampling_params)
    responses = []
    for output in outputs:
        responses.append([output.outputs[i].text for i in range(len(output.outputs))])
    return responses  









































# from vllm import LLM, SamplingParams

# def deepseek(prompt, model='deepseek-ai/DeepSeek-V2-Lite-Chat', temperature=0.7, max_tokens=512, n=1, stop=None):
#     # Create an LLM.
#     # llm = LLM(model=model, quantization= "fp8")
#     llm = LLM(model=model, trust_remote_code=True,)

#     # Create a sampling params object.
#     sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)  
#     outputs = []
#     responses = llm.generate([prompt], sampling_params)
#     # Print the responses.
#     for response in responses:
#         prompt = response.prompt
#         generated_text = response.outputs[0].text
#         outputs.append(generated_text)

#     return outputs 
