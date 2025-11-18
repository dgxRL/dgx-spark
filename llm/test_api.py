#!/usr/bin/env python3
"""
Simple test script for vLLM OpenAI-compatible API
Usage: python test_api.py [model_name] [port]
"""

import sys
from openai import OpenAI

# Get parameters from command line or use defaults
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
port = sys.argv[2] if len(sys.argv) > 2 else "8888"

# Create OpenAI client pointing to local vLLM server
client = OpenAI(
    base_url=f"http://localhost:{port}/v1",
    api_key="dummy"  # vLLM doesn't require authentication
)

print(f"Testing vLLM server at http://localhost:{port}")
print(f"Model: {model}")
print("-" * 60)

# Test 1: List available models
print("\n1. Checking available models...")
try:
    models = client.models.list()
    print(f"   ✓ Server is running!")
    print(f"   Available models: {[m.id for m in models.data]}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("\nMake sure the vLLM server is running:")
    print(f"   ./run.sh {model}")
    sys.exit(1)

# Test 2: Simple chat completion
print("\n2. Testing chat completion...")
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ],
        max_tokens=50,
        temperature=0.7
    )
    answer = response.choices[0].message.content
    print(f"   Question: What is the capital of France?")
    print(f"   Answer: {answer}")
    print(f"   ✓ Chat completion works!")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Streaming response
print("\n3. Testing streaming...")
try:
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        max_tokens=100,
        temperature=0.7,
        stream=True
    )
    print("   Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n   ✓ Streaming works!")
except Exception as e:
    print(f"\n   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYour vLLM server is ready to use!")
print(f"\nAPI endpoint: http://localhost:{port}/v1")
print("\nExample usage:")
print("""
from openai import OpenAI

client = OpenAI(
    base_url=f"http://localhost:{port}/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="{model}",
    messages=[{{"role": "user", "content": "Hello!"}}]
)

print(response.choices[0].message.content)
""")
