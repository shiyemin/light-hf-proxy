# light-hf-proxy
A light proxy solution for HuggingFace hub. It is supposed to work with diffusers, transformers and datasets from HuggingFace.

# Install / 安装

```bash
pip install light-hf-proxy --index-url https://pypi.org/simple/
```

# Usage / 使用

```python
# Just import light_hf_proxy before any huggingface libraries: transformers, diffusers, etc.
import light_hf_proxy

# Do whatever you normally do.
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
```

Just import light_hf_proxy and no more steps needed.

We only provide relay service for the following repos so as to support our own research:
- [LinkSoul.AI](https://huggingface.co/LinkSoul)
- [BAAI](https://huggingface.co/BAAI)
- [OpenAI](https://huggingface.co/openai)

Any other repos will fallback to [https://hf-mirror.com/](https://hf-mirror.com/), which is a mirror server maintained by [padeoe](https://gist.github.com/padeoe).

## Use private relay server or mirror

You can setup your own relay server or mirror to speedup model downloading. Though, we are not going to provide any tutorial or information about how to setup relay server or mirror.

To use private relay server, you can use this library as following:
```bash
RELAY_SERVER=https://your.relay.server python script.py
```

To use private mirror, you can use this library as following:
```bash
HF_MIRROR_URL=https://your.mirror.server python script.py
```


# 项目声明

本项目是一个开源项目，仅供学术研究使用。请注意以下声明：

本项目的目的是为了促进学术研究和技术探索。项目内容仅供研究人员和开发者参考和学习之用。

本项目不应用于商业场景或其他非学术研究的用途。禁止将本项目用于商业目的，包括但不限于商业产品、服务或任何形式的盈利活动。

本项目的代码和资源仅供个人学习和研究使用。未经许可，不得将本项目的代码、数据或任何相关内容用于其他项目或产品。

本项目的创建者和贡献者不对不正确使用本项目导致的后果负责。使用者应当承担使用本项目所产生的一切风险和责任。

请遵守适用的法律法规和伦理准则，确保在使用本项目时不会侵犯他人的权益，包括但不限于知识产权和隐私权。

如果您有任何关于本项目的疑问、建议或发现了任何违反上述声明的行为，请及时与我们联系。

过使用本项目，您表示您已阅读、理解并同意遵守上述声明。

# Project Statement

This project is an open-source project intended for academic research purposes only. Please note the following statement:

The purpose of this project is to facilitate academic research and technical exploration. The project content is intended for reference and learning purposes for researchers and developers.

This project should not be used for commercial purposes or any other non-academic research purposes. It is strictly prohibited to use this project for commercial purposes, including but not limited to commercial products, services, or any form of profit-making activities.

The code and resources of this project are intended for personal learning and research use. Without permission, the code, data, or any related content of this project should not be used for other projects or products.

The creators and contributors of this project are not responsible for any consequences resulting from the incorrect use of this project. Users should assume all risks and liabilities associated with the use of this project.

Please comply with applicable laws, regulations, and ethical guidelines to ensure that the use of this project does not infringe upon the rights of others, including but not limited to intellectual property rights and privacy rights.

If you have any questions, suggestions, or if you notice any behavior that violates the above statement regarding this project, please contact us promptly.

By using this project, you acknowledge that you have read, understood, and agreed to comply with the above statement.
