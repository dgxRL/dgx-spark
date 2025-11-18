#!/usr/bin/env python3
"""
Web-based chat interface for vLLM models using Gradio.

This script provides a user-friendly web interface for chatting with LLMs
running on a vLLM server.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dtype bfloat16

    # Then run this web interface:
    python web_chat.py --model meta-llama/Llama-2-7b-chat-hf

    # With custom port:
    python web_chat.py --model meta-llama/Llama-2-7b-chat-hf --port 7860

    # Create public sharing link:
    python web_chat.py --model meta-llama/Llama-2-7b-chat-hf --share

Access the interface at: http://localhost:7860

https://github.com/jasonacox/dgx-spark/
Date: 2025-11-16
"""
import argparse
import sys

try:
    import gradio as gr
except ImportError:
    print("Error: Gradio package not installed.")
    print("Install with: pip install gradio")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI package not installed.")
    print("Install with: pip install openai")
    sys.exit(1)


def create_chat_interface(model_name, base_url="http://localhost:8000/v1", 
                          default_system_prompt="You are a helpful AI assistant."):
    """Create a Gradio chat interface for vLLM models."""
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url
    )
    
    def chat(message, history, system_prompt, temperature, max_tokens, top_p):
        """Process chat message and return response."""
        # Convert Gradio history format to OpenAI messages format
        messages = [{"role": "system", "content": system_prompt}]
        
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        messages.append({"role": "user", "content": message})
        
        # Get streaming response
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True
            )
            
            # Stream the response
            partial_message = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    partial_message += chunk.choices[0].delta.content
                    yield partial_message
                    
        except Exception as e:
            yield f"Error: {str(e)}\n\nMake sure the vLLM server is running with the correct model."
    
    # Create custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    """
    
    # Create Gradio interface
    with gr.Blocks(title=f"Chat with {model_name}", css=custom_css) as demo:
        gr.Markdown(
            f"""
            # 🤖 Chat with {model_name}
            
            This interface connects to a vLLM server running on **{base_url}**.
            
            **Make sure the vLLM server is running before using this interface.**
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    avatar_images=(None, "🤖")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=9,
                        container=False
                    )
                    submit = gr.Button("Send", scale=1, variant="primary")
                
                with gr.Row():
                    clear = gr.Button("🗑️ Clear Chat", scale=1)
                    retry = gr.Button("🔄 Retry", scale=1)
                
                gr.Markdown(
                    """
                    ### Tips
                    - Press Enter to send your message
                    - Use Clear Chat to start a new conversation
                    - Adjust settings in the sidebar
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=default_system_prompt,
                    lines=3,
                    placeholder="Define the assistant's behavior..."
                )
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=2048,
                    value=256,
                    step=64,
                    label="Max Tokens",
                    info="Maximum length of response"
                )
                
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P",
                    info="Nucleus sampling threshold"
                )
                
                gr.Markdown(
                    f"""
                    ### 📊 Model Info
                    - **Model**: {model_name}
                    - **Server**: {base_url}
                    
                    ### 🔗 Quick Actions
                    - [vLLM Docs](https://docs.vllm.ai/)
                    - [Model Card](https://huggingface.co/{model_name})
                    """
                )
        
        # Event handlers
        def respond(message, chat_history, sys_prompt, temp, tokens, top_p_val):
            """Handle user message and get bot response."""
            bot_message = chat(message, chat_history, sys_prompt, temp, tokens, top_p_val)
            chat_history.append((message, ""))
            
            for partial in bot_message:
                chat_history[-1] = (message, partial)
                yield chat_history
        
        def retry_last(chat_history, sys_prompt, temp, tokens, top_p_val):
            """Retry the last message."""
            if not chat_history:
                return chat_history
            
            last_user_msg = chat_history[-1][0]
            chat_history = chat_history[:-1]
            
            bot_message = chat(last_user_msg, chat_history, sys_prompt, temp, tokens, top_p_val)
            chat_history.append((last_user_msg, ""))
            
            for partial in bot_message:
                chat_history[-1] = (last_user_msg, partial)
                yield chat_history
        
        # Submit events
        msg.submit(
            respond,
            [msg, chatbot, system_prompt, temperature, max_tokens, top_p],
            [chatbot]
        ).then(
            lambda: "",
            None,
            [msg]
        )
        
        submit.click(
            respond,
            [msg, chatbot, system_prompt, temperature, max_tokens, top_p],
            [chatbot]
        ).then(
            lambda: "",
            None,
            [msg]
        )
        
        # Clear event
        clear.click(
            lambda: [],
            None,
            [chatbot],
            queue=False
        )
        
        # Retry event
        retry.click(
            retry_last,
            [chatbot, system_prompt, temperature, max_tokens, top_p],
            [chatbot]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(
        description='Web chat interface for vLLM models'
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
        '--port',
        type=int,
        default=7860,
        help='Port to run Gradio interface (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public sharing link'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default='You are a helpful AI assistant.',
        help='Default system prompt'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Starting Web Chat Interface")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Server: {args.base_url}")
    print(f"Port: {args.port}")
    
    # Test connection to vLLM server
    try:
        client = OpenAI(api_key="EMPTY", base_url=args.base_url)
        models = client.models.list()
        print(f"✓ Connected to vLLM server")
    except Exception as e:
        print(f"\n❌ Warning: Could not connect to vLLM server at {args.base_url}")
        print(f"   {str(e)}")
        print("\nMake sure the vLLM server is running:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print(f"      --model {args.model} \\")
        print("      --dtype bfloat16")
        print("\nStarting web interface anyway...")
    
    # Create and launch interface
    demo = create_chat_interface(
        args.model,
        args.base_url,
        args.system_prompt
    )
    
    print(f"\n✓ Web interface starting...")
    print(f"   Local URL: http://localhost:{args.port}")
    if args.share:
        print("   Public URL will be generated...")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
