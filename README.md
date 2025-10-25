# DGX Spark AI Development Hub

A collection of projects, tools, and resources specifically designed for the **Nvidia DGX Spark** AI supercomputer platform.

## About the Nvidia DGX Spark

The **Nvidia DGX Spark** represents the cutting edge of AI computing, featuring revolutionary Grace Blackwell architecture that redefines how we approach machine learning and AI development.

### Key Specifications

- **GPU**: Grace Blackwell GB10 - Latest generation AI acceleration
- **Memory**: 128GB Unified Memory Architecture
- **CPU**: ARM-based Grace CPU for energy efficiency
- **Architecture**: Single GPU design with unified memory
- **Platform**: Ubuntu 24.04 ARM64 optimized for AI workloads

### Why DGX Spark is Revolutionary

**Unified Memory Architecture**: Unlike traditional GPU systems that require expensive CPU-GPU memory transfers, the DGX Spark's unified memory architecture allows direct access to the full 128GB from both CPU and GPU, eliminating bottlenecks and enabling larger model training.

**Energy Efficiency**: The ARM-based Grace CPU provides exceptional performance per watt, making it ideal for extended training sessions and development work.

**Simplified Development**: Single GPU design eliminates the complexity of distributed training while still providing enough memory and compute for meaningful AI research and development.

**Future-Proof**: Grace Blackwell architecture represents Nvidia's vision for the future of AI computing, making the DGX Spark an ideal platform for learning next-generation AI development techniques.

## Projects

### 🤖 NanoChat - Train Your Own LLM from Scratch

**Train a 1.9 billion parameter ChatGPT-like model on DGX Spark**

NanoChat is a complete implementation for training Large Language Models from scratch, specifically optimized for the DGX Spark's unique architecture. Based on Andrej Karpathy's educational framework, it provides a full pipeline from tokenization to a functional ChatGPT-like web interface.

#### What Makes It Special

- **Perfect for DGX Spark**: Optimized for GB10's 128GB unified memory
- **Educational Focus**: Learn how LLMs work end-to-end with real, functional results
- **Complete Pipeline**: Tokenization → Pretraining → Evaluation → Chat Interface
- **Hackable Codebase**: Minimal, clean, and fully customizable
- **Production Ready**: Deploy your own ChatGPT-like model

#### Quick Start

```bash
cd nanochat
./setup.sh      # Complete environment setup (run once)
./pretrain.sh   # Start training your LLM
```

#### What You'll Get

- A functional **1.9B parameter transformer model**
- **ChatGPT-like web interface** to interact with your model
- Performance that **outperforms GPT-2** on benchmarks
- Complete understanding of **modern LLM training pipeline**
- A model that's **fully yours** - no API dependencies

#### Training Details

- **Model Size**: 1.9 billion parameters (20 transformer layers)
- **Training Data**: 38 billion tokens automatically downloaded
- **Memory Usage**: Optimized for DGX Spark's 128GB unified memory
- **Training Time**: Varies based on desired quality (hours to days)
- **Cost**: Significantly lower than cloud training alternatives

#### Perfect for

- 🎓 **Learning** how modern LLMs actually work
- 🔬 **Research** on transformer architectures and training techniques
- 🛠️ **Experimentation** with AI model customization
- 🏗️ **Building** custom AI applications and chatbots
- 📚 **Teaching** AI/ML concepts with hands-on experience

**[→ Get started with NanoChat](./nanochat/README.md)**

---

## Getting Started with DGX Spark

### Prerequisites

- Nvidia DGX Spark system with Grace Blackwell GB10 GPU
- Ubuntu 24.04 ARM64 (typically pre-installed)
- Network connection for downloading dependencies and datasets
- Basic familiarity with Linux command line

### First Steps

1. **Clone this repository**:
   ```bash
   git clone https://github.com/jasonacox/dgx-spark.git
   cd dgx-spark
   ```

2. **Choose your project**: Start with NanoChat for a comprehensive LLM training experience

3. **Follow project-specific setup**: Each project includes detailed setup and usage instructions

## Platform Advantages

### For AI Development

- **Large Memory Capacity**: Train models that would OOM on traditional GPUs
- **Unified Memory**: Eliminate CPU-GPU transfer bottlenecks
- **Energy Efficient**: ARM architecture reduces power consumption
- **Educational Perfect**: Single GPU simplifies learning without sacrificing capability

### For Research

- **Cutting-Edge Architecture**: Experience the future of AI computing
- **Memory-Intensive Workloads**: Perfect for transformer models and large datasets
- **Simplified Debugging**: Single GPU eliminates distributed training complexity
- **ARM Ecosystem**: Gain experience with next-generation computing platforms

## Contributing

We welcome contributions that leverage the unique capabilities of the DGX Spark platform:

- **New Projects**: AI/ML projects optimized for Grace Blackwell architecture
- **Optimizations**: Improvements that take advantage of unified memory
- **Documentation**: Better guides and tutorials for DGX Spark development
- **Tools**: Utilities that simplify development on the platform

## Resources

- **[Nvidia DGX Spark Documentation](https://docs.nvidia.com/dgx/)**
- **[Grace Blackwell Architecture Guide](https://www.nvidia.com/en-us/data-center/grace-blackwell/)**
- **[ARM Development Resources](https://developer.arm.com/)**
- **[CUDA for ARM](https://developer.nvidia.com/cuda-gpus)**

## Future Projects

Coming soon to the DGX Spark development hub:

- 🔬 **Scientific Computing**: Optimized numerical computing workflows
- 🎨 **Computer Vision**: Image and video processing pipelines
- 📊 **Data Science**: Large-scale analytics and visualization tools
- 🎮 **Reinforcement Learning**: Game AI and robotics training environments

---

**Built for the future of AI computing. Optimized for Grace Blackwell. Designed for developers.**

