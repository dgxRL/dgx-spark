# Build a Docker image for vLLM
#
# Use the official vLLM Docker image as the base image
# from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags
#
#FROM nvcr.io/nvidia/vllm:25.09-py3
FROM nvcr.io/nvidia/vllm:25.11-py3

WORKDIR /workspace

# Clone vLLM and checkout a specific version
# To pin to a specific release, uncomment and modify the git checkout line:
RUN git clone https://github.com/vllm-project/vllm.git
# RUN cd vllm && git checkout v0.6.5  # Example: pin to v0.6.5 release
# RUN cd vllm && git checkout main     # Use latest main branch (default)
RUN cd vllm && \ 
    python use_existing_torch.py && \
    pip install -r requirements/build.txt && \
    pip install --no-build-isolation -e .

EXPOSE 8000

CMD ["/bin/bash"]

