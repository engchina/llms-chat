import gradio as gr

from modelscope import AutoTokenizer, AutoModel

DEFAULT_CKPT_PATH = "ZhipuAI/chatglm2-6b"
DEFAULT_REVISION = "v1.0.12"

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CKPT_PATH, trust_remote_code=True, revision=DEFAULT_REVISION)
model = AutoModel.from_pretrained(DEFAULT_CKPT_PATH, trust_remote_code=True, revision=DEFAULT_REVISION).half().cuda()
model = model.eval()


async def chat_stream(message, history):
    print(f"history: {history}")
    for response, _ in model.stream_chat(tokenizer, message, history=history):
        yield response


chatglm2_chat = gr.ChatInterface(fn=chat_stream, examples=["hello", "你好", "こんにちわ"], title="ChatGLM2-Chat")

chatglm2_chat.queue()

if __name__ == "__main__":
    chatglm2_chat.launch()
