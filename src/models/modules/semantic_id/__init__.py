"""Semantic-id model variants for the GRID pipeline.

The Encoder-Decoder variant (`SemanticIDEncoderDecoder`) is the default
TIGER backbone used in `tiger_train_flat.yaml`. The Decoder-Only variant
(`SemanticIDDecoderOnly`) implements the paper's Table 5 ablation
(replacing T5 encoder-decoder with a GPT-2 causal transformer).
"""
