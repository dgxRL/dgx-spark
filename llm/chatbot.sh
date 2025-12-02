#!/bin/bash
# Run the TinyLLM chatbot docker container
# Configuration can be updated in the CONFIG section below
#
# Chatbot details: https://github.com/jasonacox/TinyLLM/tree/main/chatbot
#
# =============================================================================
# CONFIG - Update these values to modify container behavior
# =============================================================================

IMAGE="jasonacox/chatbot"
CONTAINER="chatbot"
PORT="5000"

# API Configuration
OPENAI_API_KEY="sk-jarvis"
# Auto-detect host IP address (first non-loopback IPv4), fallback to localhost
HOST_IP=$(ip -4 addr show 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '^127\.' | head -n1)
if [ -z "$HOST_IP" ]; then
    HOST_IP="localhost"
fi
OPENAI_API_BASE="http://${HOST_IP}:8000/v1"

# LLM Models
LLM_MODEL="nanochat"
LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
LLM_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

# Other
THINK_FILTER="false"
INTENT_ROUTER="false"

# Volume mount
VOLUME_LOCAL="./.tinyllm"
VOLUME_CONTAINER="/app/.tinyllm"

# =============================================================================
# FUNCTIONS
# =============================================================================
cleanup_container() {
    echo "Stopping and removing existing $CONTAINER container..."
    docker stop $CONTAINER 2>/dev/null
    docker rm $CONTAINER 2>/dev/null
}

start_container() {
    echo "Starting chatbot container..."
    docker run -d \
        --name $CONTAINER \
        -p $PORT:5000 \
        -e OPENAI_API_KEY="$OPENAI_API_KEY" \
        -e OPENAI_API_BASE="$OPENAI_API_BASE" \
        -e PORT="$PORT" \
        -e LLM_MODEL="$LLM_MODEL" \
        -e THINK_FILTER="$THINK_FILTER" \
        -e INTENT_ROUTER="$INTENT_ROUTER" \
        -v $VOLUME_LOCAL:$VOLUME_CONTAINER \
        --restart unless-stopped \
        $IMAGE
}

view_logs() {
    echo "Printing logs (^C to quit)..."
    echo ""
    docker logs $CONTAINER -f
}

# =============================================================================
# MAIN
# =============================================================================
cleanup_container
start_container
view_logs