import gradio as gr

# from modelscope import AutoTokenizer, AutoModel, snapshot_download
#
# DEFAULT_CKPT_PATH = "ZhipuAI/chatglm3-6b"
# DEFAULT_REVISION = "master"
#
# model_dir = snapshot_download("ZhipuAI/chatglm3-6b-32k", revision="master")
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
# model = model.eval()

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-32k", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-32k", trust_remote_code=True).half().cuda()
model = model.eval()


async def chat_stream(message, history):
    print(f"message: {message}")
    print(f"history: {history}")
    for response, _ in model.stream_chat(tokenizer, message, history=history):
        yield response


chatglm3_chat = gr.ChatInterface(fn=chat_stream, examples=["hello", "你好", "こんにちわ"], title="ChatGLM3-Chat")

chatglm3_chat.queue()

if __name__ == "__main__":
    chatglm3_chat.launch()
