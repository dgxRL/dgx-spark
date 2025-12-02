from openai import OpenAI

# Connect to your local nanochat service
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Streaming response
response = client.chat.completions.create(
    model="nanochat",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print()  # Newline after streaming response

# Non-streaming response
response = client.chat.completions.create(
    model="nanochat",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=False
)

print(response.choices[0].message.content)
