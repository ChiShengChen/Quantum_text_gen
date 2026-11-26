"""Hybrid quantum-classical language models (revised experiments)."""
import os

# Generation grows sequences one token at a time; with the default caching allocator this
# fragments GPU memory (observed: 10 GB reserved for 0.5 GB allocated). Must be set before
# torch initialises CUDA; it changes memory management only, not any computation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
