import os
import huggingface_hub.utils

# Relay servers and mirror urls
RELAY_SERVER = os.environ.get("RELAY_SERVER", "https://relay.yimo.space")
RELAY_SERVER_SUPPORT_PREFIX = os.environ.get("RELAY_SERVER_SUPPORT_PREFIX", "https://huggingface.co/LinkSoul,https://huggingface.co/datasets/LinkSoul,https://huggingface.co/BAAI,https://huggingface.co/datasets/BAAI,https://huggingface.co/openai,https://cdn-lfs.huggingface.co")

HF_MIRROR_URL = os.environ.get("HF_MIRROR_URL", "https://hf-mirror.com")
HF_LFS_MIRROR_URL = os.environ.get("HF_LFS_MIRROR_URL", "https://lfs5.hf-mirror.com")

# Deinfe official urls
HF_OFFICIAL_URL = os.environ.get("HF_OFFICIAL_URL", "https://huggingface.co")
HF_LFS_OFFICIAL_URL = os.environ.get("HF_LFS_OFFICIAL_URL", "https://cdn-lfs.huggingface.co")


# Preprocess
RELAY_SERVER_SUPPORT_PREFIX = [p.lower() for p in RELAY_SERVER_SUPPORT_PREFIX.split(",")]

def patch_url(url):
    # Only patch hf and lfs urls
    if (not url.startswith(HF_OFFICIAL_URL)) and \
            (not url.startswith(HF_LFS_OFFICIAL_URL)):
        return url
    # If url match relay server prefix, use relay server
    for p in RELAY_SERVER_SUPPORT_PREFIX:
        if url.lower().startswith(p):
            return os.path.join(RELAY_SERVER, url)
    # If url match official prefix, replace it
    if url.startswith(HF_OFFICIAL_URL):
        return url.replace(HF_OFFICIAL_URL, HF_MIRROR_URL)
    # It's a bad choice to replace lfs prefix. We put it here just in case.
    if url.startswith(HF_LFS_OFFICIAL_URL):
        return url.replace(HF_LFS_OFFICIAL_URL, HF_LFS_MIRROR_URL)
    # You are not supposed to be here
    return url

def patched_http_backoff(*args, **kwargs):
    if len(args) >= 2 and isinstance(args[1], str):
        args[1] = patch_url(args[1])
    elif "url" in kwargs and isinstance(kwargs["url"], str):
        kwargs["url"] = patch_url(kwargs["url"])
    return huggingface_hub.utils._http.http_backoff(*args, **kwargs)

huggingface_hub.utils.http_backoff = patched_http_backoff


if __name__ == "__main__":
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
