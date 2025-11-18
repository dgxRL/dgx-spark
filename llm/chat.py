#!/usr/bin/env python3
"""
Simple CLI chat interface for vLLM models.

This script provides an interactive command-line chat interface that connects
to a vLLM OpenAI-compatible API server.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dtype bfloat16

    # Then run this chat client:
    python chat.py --model meta-llama/Llama-2-7b-chat-hf

    # With custom settings:
    python chat.py --model meta-llama/Llama-2-7b-chat-hf \
        --temperature 0.8 --max-tokens 512

https://github.com/jasonacox/dgx-spark/
Date: 2025-11-16
"""
import argparse
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI package not installed.")
    print("Install with: pip install openai")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Chat with vLLM models via OpenAI-compatible API'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Model name or path (e.g., meta-llama/Llama-2-7b-chat-hf)'
    )
    parser.add_argument(
        '--base-url', 
        type=str, 
        default='http://localhost:8000/v1',
        help='vLLM server URL (default: http://localhost:8000/v1)'
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--max-tokens', 
        type=int, 
        default=256,
        help='Maximum tokens to generate (default: 256)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter (default: 0.9)'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default='You are a helpful AI assistant.',
        help='System prompt for the assistant'
    )
    args = parser.parse_args()
    
    # Initialize OpenAI client pointing to vLLM server
    try:
        client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=args.base_url
        )
        
        # Test connection
        models = client.models.list()
        print(f"✓ Connected to vLLM server at {args.base_url}")
        
    except Exception as e:
        print(f"❌ Error: Could not connect to vLLM server at {args.base_url}")
        print(f"   {str(e)}")
        print("\nMake sure the vLLM server is running:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print(f"      --model {args.model} \\")
        print("      --dtype bfloat16")
        sys.exit(1)
    
    # Print welcome message
    print("\n" + "="*60)
    print("🤖 vLLM Chat Interface")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Settings: temperature={args.temperature}, max_tokens={args.max_tokens}, top_p={args.top_p}")
    print("\nCommands:")
    print("  Type your message and press Enter to chat")
    print("  Type 'exit' or 'quit' to end the conversation")
    print("  Type 'clear' to start a new conversation")
    print("  Type 'system <prompt>' to change the system prompt")
    print("  Press Ctrl+C to exit")
    print("="*60)
    print(f"\nSystem: {args.system_prompt}")
    print("\n🤖 Assistant: Ready.\n")
    
    # Initialize conversation with system prompt
    conversation = [
        {"role": "system", "content": args.system_prompt}
    ]
    
    # Interactive chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = input("\n💬 You: ").strip()
            except EOFError:
                print("\n\nGoodbye! 👋")
                break
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'clear':
                conversation = [{"role": "system", "content": args.system_prompt}]
                print("\n" + "="*60)
                print("Conversation cleared.")
                print("="*60)
                continue
            
            if user_input.lower().startswith('system '):
                new_system = user_input[7:].strip()
                if new_system:
                    args.system_prompt = new_system
                    conversation = [{"role": "system", "content": args.system_prompt}]
                    print(f"\n✓ System prompt updated: {args.system_prompt}")
                    print("Conversation cleared.")
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            # Get response from model
            try:
                print("\n🤖 Assistant: ", end="", flush=True)
                
                response = client.chat.completions.create(
                    model=args.model,
                    messages=conversation,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    stream=True  # Enable streaming for real-time output
                )
                
                # Stream the response
                assistant_message = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        assistant_message += content
                
                print()  # New line after response
                
                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": assistant_message})
                
            except Exception as e:
                print(f"\n❌ Error generating response: {str(e)}")
                # Remove the user message that caused the error
                conversation.pop()
            
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
