import os

import gradio as gr
import cohere
from cohere.responses.classify import Example
from dotenv import load_dotenv, find_dotenv

# read local .env file
_ = load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

chat_history = []


async def chat_stream(question, history, model, citation_quality, prompt_truncation, randomness):
    if question is None or len(question) == 0:
        return
    history = chat_history
    async with cohere.AsyncClient(api_key=COHERE_API_KEY) as aio_co:
        streaming_chat = await aio_co.chat(
            message=question,
            chat_history=history,
            model=model,
            stream=True,
            citation_quality=citation_quality,
            prompt_truncation=prompt_truncation,
            temperature=randomness
        )
        completion = ""
        async for token in streaming_chat:
            # print(f"chat_stream token: {token}")
            # print(f"chat_stream type(token): {type(token)}")
            if isinstance(token, cohere.responses.chat.StreamTextGeneration):
                completion += token.text
                yield completion

        if len(chat_history) == 10:
            chat_history.pop(0)
        user_message = {"user_name": "User", "text": question}
        bot_message = {"user_name": "Chatbot", "text": completion}

        chat_history.append(user_message)
        chat_history.append(bot_message)


model_text = gr.Dropdown(label="Model",
                         choices=["command", "command-nightly",
                                  "command-light", "command-light-nightly"],
                         value="command",
                         )

citation_quality_text = gr.Dropdown(label="Citation Quality",
                                    choices=["accurate", "fast"],
                                    value="accurate",
                                    interactive=True
                                    )

prompt_truncation_text = gr.Dropdown(label="Prompt Truncation",
                                     choices=["auto", "off"],
                                     value="auto",
                                     interactive=True
                                     )

randomness_text = gr.Slider(label="Randomness(Temperature)",
                            minimum=0,
                            maximum=2,
                            step=0.1,
                            value=0.3,
                            interactive=True
                            )

demo = gr.ChatInterface(fn=chat_stream,
                        additional_inputs=[model_text,
                                           citation_quality_text,
                                           prompt_truncation_text,
                                           randomness_text
                                           ],
                        additional_inputs_accordion_name="Additional Inputs",
                        examples=[
                            ["Can you give me a global market overview of the solar panels?", "command", "accurate",
                             "auto", 0.3],
                            ["Gather business intelligence on the Chinese markets", "command-nightly", "accurate",
                             "auto", 0.3],
                            ["Summarize recent news about the North American tech job market", "command-light", "fast",
                             "off", 0.9],
                            ["Give me a rundown of AI startups in the productivity space", "command-light-nightly",
                             "fast",
                             "off", 2]],
                        title="(Unofficial) Chat with Cohere Coral")

demo.queue()
if __name__ == "__main__":
    demo.launch()
