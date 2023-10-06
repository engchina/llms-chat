# llm-chat-web-ui
LLMs Chat with Web UI

## prepare

create conda environment,

```
conda create -n llm-chat-web-ui python=3.9 -y
conda activate llm-chat-web-ui
```

install pytorch,

```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

install requirements,

```
pip install -r requirements.txt
```

(optional)install flash-attn,

```
git clone https://github.com/Dao-AILab/flash-attention; cd flash-attention
pip install flash-attn --no-build-isolation
pip install csrc/layer_norm
pip install csrc/rotary
```

## chat with qwen

```
python qwen_chat.py
```

open your browser and access [http://localhost:7860](http://localhost:7860)

## chat with chatglm2

```
python chatglm2_chat.py
```

open your browser and access [http://localhost:7860](http://localhost:7860)


## chat with cohere coral

create .env file,

```
cp .env.example .env
```

modify cohere api key,

```
vi .env

---
COHERE_API_KEY=<input your cohere api key which is from https://dashboard.cohere.com/api-keys>
---
```

launch cohere coral,

```
python cohere_coral_chat.py
```

open your browser and access [http://localhost:7860](http://localhost:7860)
