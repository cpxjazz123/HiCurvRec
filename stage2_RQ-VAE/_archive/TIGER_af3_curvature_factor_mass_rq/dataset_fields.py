"""UNUSED — leftover of the removed plug-in host; slated for deletion.

The trainer is byte-identical to the frozen baseline except ``ARM_NAME``, so batches
stay ``TensorDataset(train_embeddings)`` and carry no fields. Nothing imports this
module; it is kept only because this step cannot remove files.
"""
