# AttentionIsAllYouNeed

A complete from-scratch PyTorch implementation of the Transformer model from the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017).

## Overview

This repository contains a **completely from-scratch implementation** of the Transformer architecture, built without relying on existing transformer libraries. Every component has been implemented from the ground up, including:

- **Custom Transformer Architecture**: Full encoder-decoder implementation with multi-head attention
- **Custom BPE Tokenizer**: Built-from-scratch Byte-Pair Encoding tokenizer for text preprocessing
- **WMT Dataset Integration**: Trained and tested on the WMT (Workshop on Machine Translation) dataset

This project demonstrates a deep understanding of the Transformer architecture by implementing every single component without external transformer libraries.

## Key Features

- **Complete From-Scratch Implementation**: No external transformer libraries used - every component built independently
- **Custom BPE Tokenizer**: Implemented Byte-Pair Encoding tokenizer from scratch for efficient text processing
- **Multi-Head Self-Attention**: Core attention mechanism allowing the model to focus on different sequence parts
- **Positional Encoding**: Sinusoidal positional embeddings for sequence order information
- **Encoder-Decoder Architecture**: Full implementation of both encoder and decoder stacks
- **Layer Normalization**: Pre-normalization approach for stable training
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Residual Connections**: Skip connections for improved gradient flow
- **WMT Dataset Training**: Trained and evaluated on Workshop on Machine Translation datasets

