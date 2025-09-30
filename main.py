from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
import os
import time
import psutil
from datetime import datetime
from typing import Dict, List
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

# Metrics tracking
response_times: List[float] = []
metrics_lock = asyncio.Lock()

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

def calculate_percentiles(times: List[float]) -> Dict[str, float]:
    """Calculate P50, P90, P95, P99 percentiles"""
    if not times:
        return {"P50": 0, "P90": 0, "P95": 0, "P99": 0}
    
    sorted_times = sorted(times)
    n = len(sorted_times)
    
    def percentile(p: float) -> float:
        index = (p / 100) * (n - 1)
        if index.is_integer():
            return sorted_times[int(index)]
        else:
            lower = sorted_times[int(index)]
            upper = sorted_times[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    return {
        "P50": percentile(50),
        "P90": percentile(90),
        "P95": percentile(95),
        "P99": percentile(99)
    }

async def log_response_time(response_time: float):
    """Log response time for metrics calculation"""
    async with metrics_lock:
        response_times.append(response_time)
        # Keep only last 1000 measurements to prevent memory issues
        if len(response_times) > 1000:
            response_times.pop(0)

def streaming_data_generator(content):
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
        yield f"data: {json.dumps(chunk)}\n\n"
    
    yield f"data: [DONE]\n\n"

async def handle_completion(request: Request, model: str = None, tracing_enabled: bool = False, trace_id: str = None):
    """Handle completion requests with tracing and metrics"""
    start_time = time.time()
    data = await request.json()
    requested_model = data.get("model") or model or "gpt-3.5-turbo"
    
    # Log tracing information if enabled
    if tracing_enabled:
        print(f"[TRACE] {trace_id}: Processing request for model {requested_model}")
    
    # add latency if configured
    if LATENCY > 0:
        await asyncio.sleep(LATENCY)
    
    content = get_response_content()
    
    # Calculate response time
    response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    await log_response_time(response_time)
    
    if tracing_enabled:
        print(f"[TRACE] {trace_id}: Response generated in {response_time:.2f}ms")
    
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

# Chat completions endpoint supporting both OpenAI and Azure formats
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/openai/deployments/{model:path}/chat/completions")  # azure compatible endpoint
async def completion(request: Request, model: str = None):
    return await handle_completion(request, model)

# Gateway endpoints for testing scenarios
@app.post("/gateway/v1/chat/completions")
async def gateway_completion(request: Request):
    """Gateway endpoint with tracing support"""
    tracing_enabled = request.headers.get("X-Tracing-Enabled", "").lower() == "true"
    trace_id = request.headers.get("X-Trace-ID", "no-trace-id")
    
    return await handle_completion(request, tracing_enabled=tracing_enabled, trace_id=trace_id)

# Metrics endpoint for monitoring
@app.get("/metrics")
async def get_metrics():
    """Get current metrics including CPU usage and response time percentiles"""
    async with metrics_lock:
        current_response_times = response_times.copy()
    
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    # Calculate percentiles
    percentiles = calculate_percentiles(current_response_times)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage_percent": cpu_percent,
        "memory_usage_percent": memory_percent,
        "total_requests": len(current_response_times),
        "response_time_percentiles_ms": percentiles,
        "average_response_time_ms": sum(current_response_times) / len(current_response_times) if current_response_times else 0,
        "min_response_time_ms": min(current_response_times) if current_response_times else 0,
        "max_response_time_ms": max(current_response_times) if current_response_times else 0
    }
    
    return metrics

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
