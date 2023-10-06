import gradio as gr
import torch

from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

DEFAULT_CKPT_PATH = "baichuan-inc/Baichuan2-13B-Chat-4bits"
DEFAULT_REVISION = "v1.0.2"

model_dir = snapshot_download(DEFAULT_CKPT_PATH, revision=DEFAULT_REVISION)
# tokenizer = AutoTokenizer.from_pretrained(
#     model_dir, trust_remote_code=True, resume_download=True,
# )
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     device_map="cuda",
#     trust_remote_code=True,
#     resume_download=True,
# ).eval()
#
# config = GenerationConfig.from_pretrained(
#     model_dir, trust_remote_code=True, resume_download=True,
# )

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True,
                                          resume_download=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", torch_dtype=torch.bfloat16,
                                             trust_remote_code=True, resume_download=True)
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, resume_download=True, )


async def chat_stream(message, history):
    print(f"history: {history}")
    messages = history
    messages.append({"role": "user", "content": message})
    print(f"messages: {messages}")
    for response in model.chat(tokenizer, messages, stream=True):
        yield response


# TODO "Expected a list of lists or list of tuples. Received: {message_pair}"
# gradio and baichuan2 message format is conflit
baichuan2_chat = gr.ChatInterface(fn=chat_stream, examples=["hello", "你好", "こんにちわ"], title="Baichuan2-Chat")

if __name__ == "__main__":
    baichuan2_chat.queue().launch()
