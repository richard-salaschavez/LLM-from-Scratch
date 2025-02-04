# Transformer-Based LLM from Scratch 🧠⚡

**A PyTorch implementation of a decoder-only Transformer for text generation, designed to demystify modern LLM architectures.**

---

## Overview  
This project implements a **Transformer-based Large Language Model (LLM)** from scratch using PyTorch, trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. It serves as a hands-on exploration of the Transformer architecture, covering core components like self-attention, positional embeddings, and autoregressive generation. Ideal for ML engineers seeking to deepen their understanding of LLM internals and scalable model design.

---

## Features  
- ✅ **Decoder-only Transformer** with configurable hyperparameters (`n_layer=5`, `n_head=6`, `n_embd=256`).  
- ✅ **Multi-head self-attention** for parallelized context modeling.  
- ✅ **Positional embeddings** to encode token order.  
- ✅ **Layer normalization** and **dropout** (`0.2`) for stable training.  
- ✅ **Autoregressive text generation** with temperature-free sampling.  
- ✅ GPU-accelerated training via CUDA and PyTorch optimizations.  

---

## Technologies Used  
- **PyTorch** (Model architecture, training pipeline, tensor ops)  
- **CUDA** (GPU acceleration)  
- **Python** (Data preprocessing, tokenization)  

---

## Key Learnings  
Through this project, I gained practical insights into:  
- **Self-Attention Mechanisms**: Scaled dot-product attention for token relationship modeling.  
- **Multi-Head Attention**: Parallel attention heads to capture diverse contextual patterns.  
- **Positional Encoding**: Learnable embeddings to inject sequence order into token representations.  
- **Training Optimization**: Hyperparameter tuning (batch size, learning rate), AdamW optimizer, and loss convergence strategies.  
- **Layer Normalization & Dropout**: Techniques to stabilize deep networks and mitigate overfitting.  
- **End-to-End Pipeline Design**: From raw text tokenization to model deployment.  

---
