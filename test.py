import light_hf_proxy

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = "LinkSoul/Chinese-Llama-2-7b"
cache_dir = "cache"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir).half().cuda()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

prompt = instruction.format("用中文回答，When is the best time to visit Beijing, and do you have any suggestions for me?")
generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)
