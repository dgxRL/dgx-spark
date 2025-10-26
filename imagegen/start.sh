#!/bin/bash

# ImageGen Start Script for Nvidia DGX Spark
# This script will launch ComfyUI interface optimized for Grace Blackwell GB10
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

set -e  # Exit on any error

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[COMFYUI]${NC} $1"
}

# Default configuration
PORT=8188
HOST="0.0.0.0"
DAEMON_MODE=false
LOG_FILE=""
EXTRA_ARGS=""

# Function to show help
show_help() {
    print_header "ComfyUI Launcher for DGX Spark Grace Blackwell GB10"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT        Port to run ComfyUI on (default: 8188)"
    echo "  -h, --host HOST        Host to bind to (default: 0.0.0.0)"
    echo "  -d, --daemon           Run as daemon in background"
    echo "  -l, --log FILE         Log output to file (only with --daemon)"
    echo "  -s, --stop             Stop running daemon"
    echo "  -r, --restart          Restart daemon"
    echo "  --status               Check daemon status"
    echo "  --cpu                  Force CPU mode (not recommended)"
    echo "  --low-vram             Use low VRAM mode"
    echo "  --high-vram            Use high VRAM mode (default for DGX Spark)"
    echo "  --fp32                 Use FP32 instead of FP16"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Start ComfyUI normally"
    echo "  $0 -d -l comfyui.log   # Start as daemon with logging"
    echo "  $0 -p 8080             # Start on port 8080"
    echo "  $0 --stop              # Stop daemon"
    echo "  $0 --status            # Check if daemon is running"
    echo ""
    echo "DGX Spark optimizations:"
    echo "  • Automatically detects Grace Blackwell GB10"
    echo "  • Uses FP16 for memory efficiency"
    echo "  • Optimizes for 128GB unified memory"
    echo "  • Enables high VRAM mode by default"
    echo ""
}

# Function to check if ComfyUI is set up
check_setup() {
    if [ ! -d "ComfyUI" ]; then
        print_error "ComfyUI not found. Please run ./setup.sh first."
        exit 1
    fi
    
    if [ ! -d "comfyui-env" ]; then
        print_error "Python virtual environment not found. Please run ./setup.sh first."
        exit 1
    fi
    
    if [ ! -f "launch_comfyui.sh" ]; then
        print_error "Launch script not found. Please run ./setup.sh first."
        exit 1
    fi
}

# Function to check if any models are installed
check_models() {
    local model_count=0
    model_count=$((model_count + $(find models/checkpoints/ -name "*.safetensors" -o -name "*.ckpt" 2>/dev/null | wc -l)))
    
    if [ $model_count -eq 0 ]; then
        print_warning "No models found in models/checkpoints/"
        print_warning "You may want to run: ./models.sh install base"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Please install models first with: ./models.sh install base"
            exit 1
        fi
    else
        print_status "Found $model_count model(s) ready to use"
    fi
}

# Function to get PID of running ComfyUI daemon
get_daemon_pid() {
    pgrep -f "python.*main.py.*--listen" 2>/dev/null | head -1
}

# Function to check daemon status
daemon_status() {
    local pid=$(get_daemon_pid)
    if [ -n "$pid" ]; then
        print_status "ComfyUI daemon is running (PID: $pid)"
        print_status "Access at: http://$(hostname -I | awk '{print $1}'):$PORT"
        return 0
    else
        print_warning "ComfyUI daemon is not running"
        return 1
    fi
}

# Function to stop daemon
stop_daemon() {
    local pid=$(get_daemon_pid)
    if [ -n "$pid" ]; then
        print_status "Stopping ComfyUI daemon (PID: $pid)..."
        kill "$pid"
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            print_warning "Process still running, force killing..."
            kill -9 "$pid"
        fi
        print_status "ComfyUI daemon stopped"
    else
        print_warning "No running ComfyUI daemon found"
    fi
}

# Function to start ComfyUI
start_comfyui() {
    # Setup DGX Spark environment variables
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH=/usr/local/cuda-13.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
    
    # Optimize for Grace Blackwell unified memory
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
    export CUDA_LAUNCH_BLOCKING=0
    
    # Activate virtual environment
    print_status "Activating Python virtual environment..."
    source comfyui-env/bin/activate
    
    # Verify CUDA
    print_status "Verifying CUDA availability..."
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    
    # Build command arguments
    local args=(
        "--listen" "$HOST"
        "--port" "$PORT"
    )
    
    # DGX Spark optimizations
    if [[ ! "$EXTRA_ARGS" =~ --cpu ]]; then
        args+=("--fp16-vae")
        if [[ ! "$EXTRA_ARGS" =~ --low-vram ]]; then
            args+=("--normalvram")  # Use normal VRAM mode for 128GB memory
        fi
    fi
    
    # Add any extra arguments
    if [ -n "$EXTRA_ARGS" ]; then
        args+=($EXTRA_ARGS)
    fi
    
    # Change to ComfyUI directory
    cd ComfyUI/
    
    # Display startup information
    print_header "Starting ComfyUI with DGX Spark optimizations..."
    echo ""
    print_status "Configuration:"
    print_status "  Host: $HOST"
    print_status "  Port: $PORT"
    print_status "  Mode: $([ "$DAEMON_MODE" = true ] && echo "Daemon" || echo "Interactive")"
    print_status "  GPU: Grace Blackwell GB10 optimization enabled"
    print_status "  Memory: 128GB unified memory optimized"
    echo ""
    
    if [ "$DAEMON_MODE" = true ]; then
        print_status "Starting ComfyUI as daemon..."
        if [ -n "$LOG_FILE" ]; then
            nohup python main.py "${args[@]}" > "../$LOG_FILE" 2>&1 &
            print_status "Logging to: $LOG_FILE"
        else
            nohup python main.py "${args[@]}" > /dev/null 2>&1 &
        fi
        local pid=$!
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            print_status "ComfyUI daemon started successfully (PID: $pid)"
        else
            print_error "Failed to start ComfyUI daemon"
            exit 1
        fi
    else
        print_status "Starting ComfyUI in interactive mode..."
        print_status "Press Ctrl+C to stop"
        echo ""
        python main.py "${args[@]}"
    fi
    
    echo ""
    print_status "ComfyUI is available at: http://$(hostname -I | awk '{print $1}'):$PORT"
    
    if [ "$DAEMON_MODE" = true ]; then
        echo ""
        print_status "Daemon management commands:"
        print_status "  Check status: ./start.sh --status"
        print_status "  Stop daemon:  ./start.sh --stop"
        print_status "  Restart:      ./start.sh --restart"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -d|--daemon)
            DAEMON_MODE=true
            shift
            ;;
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        -s|--stop)
            check_setup
            stop_daemon
            exit 0
            ;;
        -r|--restart)
            check_setup
            stop_daemon
            sleep 1
            DAEMON_MODE=true
            break
            ;;
        --status)
            daemon_status
            exit $?
            ;;
        --cpu)
            EXTRA_ARGS="$EXTRA_ARGS --cpu"
            shift
            ;;
        --low-vram)
            EXTRA_ARGS="$EXTRA_ARGS --lowvram"
            shift
            ;;
        --high-vram)
            EXTRA_ARGS="$EXTRA_ARGS --normalvram"
            shift
            ;;
        --fp32)
            EXTRA_ARGS="$EXTRA_ARGS --force-fp32"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run pre-flight checks
check_setup
check_models

# Start ComfyUI
start_comfyui

