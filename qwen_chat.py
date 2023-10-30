import gradio as gr

from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope import GenerationConfig

DEFAULT_CKPT_PATH = "Qwen/Qwen-14B-Chat-Int8"
DEFAULT_REVISION = "v1.0.6"

model_dir = snapshot_download(DEFAULT_CKPT_PATH, revision=DEFAULT_REVISION)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, trust_remote_code=True, resume_download=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda",
    trust_remote_code=True,
    resume_download=True,
).eval()

config = GenerationConfig.from_pretrained(
    model_dir, trust_remote_code=True, resume_download=True,
)


async def chat_stream(message, history):
    for response in model.chat_stream(tokenizer, message, history=history, generation_config=config):
        yield response


qwen_chat = gr.ChatInterface(fn=chat_stream, examples=["hello", "你好", "こんにちわ"], title="Qwen-Chat")

qwen_chat.queue()

if __name__ == "__main__":
    qwen_chat.launch()
