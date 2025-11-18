FROM nvcr.io/nvidia/vllm:25.09-py3

WORKDIR /workspace

RUN git clone https://github.com/vllm-project/vllm.git
RUN cd vllm && \ 
    python use_existing_torch.py && \
    pip install -r requirements/build.txt && \
    pip install --no-build-isolation -e .

EXPOSE 8000

CMD ["/bin/bash"]

