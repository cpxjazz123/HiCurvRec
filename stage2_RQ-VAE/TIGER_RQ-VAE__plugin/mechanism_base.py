"""The one file a mechanism is allowed to add (``mechanism.py`` learns from this).

A candidate tree is a copy of this baseline plus exactly one new file, ``mechanism.py``.
Everything else — paths, hyper-parameters, DDP, the data pipeline, SID export — is
host-owned. The trainer probes these optional hooks:

``runtime_context.item_ids()``
    The host always carries item ids with each batch and publishes the row identity
    through ``runtime_context``: any mechanism file may read it directly. ``set_context``
    below is only for mechanisms that want to pre-compute per-batch quantities.

``prepare(train_ids, all_ids) -> dict``
    Called once before training (rank 0) to build frozen tables / fit constants.
    The returned dict is written to ``training_metrics.jsonl`` as ``event=mechanism``.

``set_context(item_ids, training: bool) -> None``
    Called before every forward (train) and before the export/eval inference pass.
    A stateful distance callable reads the current ids from here; the model stays
    untouched, which is what keeps the candidate to one file.

``DISTANCE = {"name": str, "function": callable, "description": str}``
    Registered lazily by ``distances.build_distance`` when the mechanism names it.

``readouts() -> dict`` / ``switches() -> dict``
    Read-only diagnostics merged into ``train_end`` / ``mechanism`` records.

Minimal skeleton::

    needs_item_ids = True
    DISTANCE = {"name": "my_cost", "function": _cost, "description": "..."}

    def prepare(train_ids, all_ids):
        return {"items": int(len(all_ids))}

    def set_context(item_ids, training):
        global _current_ids
        _current_ids = item_ids

    def _cost(latent, codebook):
        # latent rows correspond to _current_ids
        ...

    def readouts():
        return {}

    def switches():
        return {"behaviour_off": False, "geometry_off": False}
"""
