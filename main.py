from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
import os
import time
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TOKEN_COUNT = int(os.getenv("TOKEN_COUNT", "100"))
LATENCY = float(os.getenv("LATENCY", "0"))

BASE_SENTENCE = """
This is a comprehensive exploration of artificial intelligence and machine learning concepts that demonstrates the intricate relationships between various computational methodologies and their practical applications in modern technology systems. The field encompasses numerous sophisticated algorithms including neural networks, deep learning architectures, natural language processing frameworks, computer vision systems, reinforcement learning paradigms, and statistical modeling approaches that collectively form the foundation of contemporary AI research and development. These interconnected disciplines leverage mathematical principles from linear algebra, calculus, probability theory, information theory, and optimization techniques to create intelligent systems capable of learning, reasoning, and adapting to complex environments. The evolution of these technologies has been driven by advances in computational power, data availability, algorithmic innovations, and theoretical breakthroughs that have enabled the development of increasingly sophisticated models and applications. Machine learning algorithms such as supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning each offer unique advantages for different types of problems and data characteristics. Deep learning architectures including convolutional neural networks, recurrent neural networks, transformer models, generative adversarial networks, and variational autoencoders have revolutionized numerous domains including image recognition, natural language understanding, speech processing, and autonomous systems. The training processes for these models involve complex optimization procedures using gradient descent, backpropagation, regularization techniques, and various loss functions to minimize prediction errors and improve generalization capabilities. Data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation methodologies are crucial components of the machine learning pipeline that directly impact the performance and reliability of AI systems. The deployment of machine learning models in production environments requires careful consideration of scalability, latency, reliability, security, and interpretability factors to ensure robust and trustworthy AI applications. Ethical considerations in AI development include fairness, accountability, transparency, privacy protection, and bias mitigation strategies that are essential for responsible AI deployment in real-world scenarios. The interdisciplinary nature of AI research combines insights from computer science, mathematics, statistics, cognitive science, neuroscience, and domain-specific expertise to advance our understanding of intelligent systems and their capabilities. Emerging trends in AI include federated learning, edge computing, quantum machine learning, neuromorphic computing, and hybrid AI systems that integrate symbolic reasoning with neural approaches to address complex problem-solving challenges. The societal implications of AI technologies extend beyond technical considerations to encompass economic impacts, workforce transformation, educational requirements, policy development, and international cooperation frameworks that shape the future of human-AI interaction and collaboration in various sectors of society.
"""

def get_response_content():
    """Generate response content based on TOKEN_COUNT environment variable"""
    words = BASE_SENTENCE.strip().split()
    estimated_tokens = len(words)
    
    if TOKEN_COUNT > estimated_tokens:
        repetitions = (TOKEN_COUNT // estimated_tokens) + 1
        content = (BASE_SENTENCE.strip() + " ") * repetitions
        content_words = content.split()
        content = " ".join(content_words[:TOKEN_COUNT])
    else:
        content = " ".join(words[:TOKEN_COUNT])
    
    return content

async def streaming_data_generator(content):
    """Generate a streaming response with the given content"""
    response_id = uuid.uuid4().hex
    words = content.split(" ")
    for word in words:
        word = word + " "
        chunk = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {"content": word}}],
        }
        sleep_time = os.environ.get("SLEEP_TIME", "0.1")
        await asyncio.sleep(float(sleep_time))
        yield f"data: {json.dumps(chunk)}\n\n"

    # # return usage stats
    # yield (
    # "data: "
    # f'{{"id": "fake-id", "object": "chat.completion", "choices": [], '
    # f'"usage": {{"prompt_tokens": {TOKEN_COUNT}, '
    # f'"completion_tokens": {TOKEN_COUNT}, '
    # f'"total_tokens": {TOKEN_COUNT}}}}}\n\n'
    # )    
    yield f"data: [DONE]\n\n"

async def streaming_text_generator(content):
    """Generate a streaming response for text completions"""
    response_id = uuid.uuid4().hex
    words = content.split(" ")
    for word in words:
        word = word + " "
        chunk = {
            "id": f"cmpl-{response_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"index": 0, "text": word, "logprobs": None, "finish_reason": None}],
        }
        sleep_time = os.environ.get("SLEEP_TIME", "0.1")
        await asyncio.sleep(float(sleep_time))
        yield f"data: {json.dumps(chunk)}\n\n"

    # return usage stats
    yield (
    "data: "
    f'{{"id": "fake-id", "object": "text_completion", "choices": [], '
    f'"usage": {{"prompt_tokens": {TOKEN_COUNT}, '
    f'"completion_tokens": {TOKEN_COUNT}, '
    f'"total_tokens": {TOKEN_COUNT}}}}}\n\n'
    )    
    yield f"data: [DONE]\n\n"

# Chat completions endpoint supporting both OpenAI and Azure formats
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/openai/deployments/{model:path}/chat/completions")  # azure compatible endpoint
async def completion(request: Request, model: str = None):
    data = await request.json()
    requested_model = data.get("model") or model or "gpt-3.5-turbo"
    
    # add latency if configured
    if LATENCY > 0:
        await asyncio.sleep(LATENCY)
    
    content = get_response_content()
    
    if data.get("stream") == True:
        return StreamingResponse(
            content=streaming_data_generator(content),
            media_type="text/event-stream",
        )
    else:
        response_id = uuid.uuid4().hex
        response = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "system_fingerprint": "fp_" + uuid.uuid4().hex[:12],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": TOKEN_COUNT,
                "total_tokens": 10 + TOKEN_COUNT
            },
        }
        return response

# Text completions endpoint supporting both OpenAI and Azure formats
@app.post("/completions")
@app.post("/v1/completions")
@app.post("/openai/deployments/{model:path}/completions")  # azure compatible endpoint
async def text_completion(request: Request, model: str = None):
    data = await request.json()
    requested_model = data.get("model") or model or "gpt-3.5-turbo-instruct"
    
    # add latency if configured
    if LATENCY > 0:
        await asyncio.sleep(LATENCY)
    
    content = get_response_content()
    
    if data.get("stream") == True:
        return StreamingResponse(
            content=streaming_text_generator(content),
            media_type="text/event-stream",
        )
    else:
        response_id = uuid.uuid4().hex
        response = {
            "id": f"cmpl-{response_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "text": content,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": TOKEN_COUNT,
                "total_tokens": 10 + TOKEN_COUNT
            },
        }
        return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
