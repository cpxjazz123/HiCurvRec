"""Quantized geometry fidelity against the frozen graph curvature (issue #249, step 6).

The graph side is the frozen #248 target: Eq. A.1 with hop distances on the undirected projection
of the training-visible graph, over the centre's whole component as the reference set.

The quantized side applies the **same formula** with the same reference indices, but with the
distances of the space the quantized vectors live in. The two sides therefore differ in their
distance function and in their scale; the scale-free headline is the sign agreement.

The hard SID is decoded with the codebook of the **same** checkpoint generation; the continuous
``z`` is reported separately and never substitutes for the discrete SID.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from tools.graph_infra import fidelity
from tools.graph_infra.diff_sectional import build_metric

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / 'assets'
ARM_DIR = REPO / 'stage2_RQ-VAE' / 'baseline' / 'TIGER_RQ-VAE'
RESULT_DIR = REPO / 'results' / 'stage2_RQ-VAE' / 'TIGER_RQ-VAE'
REEXPORT_DIR = REPO / 'results' / 'research' / 'infra246' / 'tiger_reexport'
EMBEDDINGS = REPO / 'stage1_GeneEmbedding' / 'output' / 'sentence_t5.npy'


def load_codebooks(checkpoint_path: Path) -> list[np.ndarray]:
    """The three level codebooks of a checkpoint, in level order."""
    blob = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = blob['state_dict']
    codebooks = []
    for level in range(3):
        key = f'rq.vq_layers.{level}.embed.weight'
        if key not in state:
            raise KeyError(f'no codebook for level {level} in {checkpoint_path}')
        codebooks.append(state[key].detach().cpu().numpy().astype(np.float64))
    return codebooks


def load_continuous_z(checkpoint_path: Path) -> np.ndarray:
    """The encoder output ``z`` of the frozen checkpoint, read-only, through the arm's own code."""
    import sys
    sys.path.insert(0, str(ARM_DIR))
    from model import RQVAE  # noqa: E402  (the frozen arm's own model)
    import train_rqvae  # noqa: E402

    config = train_rqvae._tokenizer_config()
    model = RQVAE(config, in_dim=768, distance=train_rqvae.DISTANCE)
    blob = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(blob['state_dict'])
    model.eval()
    embeddings = np.load(EMBEDDINGS).astype(np.float32)
    with torch.no_grad():
        encoded = model.encoder(torch.from_numpy(embeddings))
    return encoded.detach().cpu().numpy().astype(np.float64)


def load_sid_table(path: Path) -> np.ndarray:
    table = json.loads(Path(path).read_text())
    keys = sorted(table, key=int)
    return np.array([table[key] for key in keys], dtype=np.int64)


def sample_triples(reader, items: int, seed: int = 249002) -> np.ndarray:
    """A deterministic sample of triples, spread over distinct centres."""
    rng = np.random.default_rng(seed)
    chosen = []
    centres = np.flatnonzero(reader.item_count > 0)
    order = rng.permutation(len(centres))
    for index in order:
        item = int(centres[index])
        rows = reader.item_rows(item)
        if not len(rows['b']):
            continue
        position = int(rng.integers(0, len(rows['b'])))
        chosen.append([item, int(rows['b'][position]), int(rows['c'][position])])
        if len(chosen) >= items:
            break
    return np.array(chosen, dtype=np.int64)


def evaluate(triples: np.ndarray, reader, node_moments: dict, undirected,
             q_vectors: np.ndarray, z_vectors: np.ndarray,
             metric_family: str = 'euclidean', kappa: float = 1.0,
             scale: float = 1.0) -> dict:
    """Evaluate the sampled triples on both sides and compare."""
    from scipy.sparse.csgraph import shortest_path
    metric = build_metric(metric_family, kappa, scale)
    labels = node_moments['component']
    graph_values = []
    quantized_values = []
    continuous_values = []
    graph_distances = []
    quantized_distances = []
    for m, b, c in triples.tolist():
        references = np.flatnonzero(labels == labels[m])
        source = shortest_path(undirected, directed=False, unweighted=True,
                               indices=[m, b, c])
        d_am = source[0][references]
        d_ab = source[1][references]
        d_ac = source[2][references]
        d_bc = float(shortest_path(undirected, directed=False, unweighted=True,
                                   indices=[b])[0][c])
        graph_values.append(float(np.mean(d_am ** 2 + d_bc ** 2 / 4.0
                                         - (d_ab ** 2 + d_ac ** 2) / 2.0)))
        q_reference = torch.from_numpy(q_vectors[references])
        q_distances = {
            'd_am': metric(q_reference, torch.from_numpy(q_vectors[m]).unsqueeze(0)),
            'd_ab': metric(q_reference, torch.from_numpy(q_vectors[b]).unsqueeze(0)),
            'd_ac': metric(q_reference, torch.from_numpy(q_vectors[c]).unsqueeze(0)),
            'd_bc': metric(torch.from_numpy(q_vectors[b]), torch.from_numpy(q_vectors[c])),
        }
        quantized_values.append(float(torch.mean(
            q_distances['d_am'] ** 2 + q_distances['d_bc'] ** 2 / 4.0
            - (q_distances['d_ab'] ** 2 + q_distances['d_ac'] ** 2) / 2.0)))
        z_reference = torch.from_numpy(z_vectors[references])
        z_distances = {
            'd_am': metric(z_reference, torch.from_numpy(z_vectors[m]).unsqueeze(0)),
            'd_ab': metric(z_reference, torch.from_numpy(z_vectors[b]).unsqueeze(0)),
            'd_ac': metric(z_reference, torch.from_numpy(z_vectors[c]).unsqueeze(0)),
            'd_bc': metric(torch.from_numpy(z_vectors[b]), torch.from_numpy(z_vectors[c])),
        }
        continuous_values.append(float(torch.mean(
            z_distances['d_am'] ** 2 + z_distances['d_bc'] ** 2 / 4.0
            - (z_distances['d_ab'] ** 2 + z_distances['d_ac'] ** 2) / 2.0)))
        graph_distances.append(d_am)
        quantized_distances.append(q_distances['d_am'].detach().numpy().ravel())
    graph_values = np.asarray(graph_values)
    quantized_values = np.asarray(quantized_values)
    continuous_values = np.asarray(continuous_values)
    # the graph side recomputed here must equal the frozen #248 target
    frozen = []
    for m, b, c in triples.tolist():
        rows = reader.item_rows(m)
        order = np.lexsort((rows['c'], rows['b']))
        keys = rows['b'][order] * (1 << 32) + rows['c'][order]
        position = int(np.searchsorted(keys, b * (1 << 32) + c))
        frozen.append(float(rows['curvature'][order[position]]))
    frozen = np.asarray(frozen)
    aligned = fidelity.scale_factor(graph_values, quantized_values)
    report = {
        'metric_family': metric_family,
        'metric_kappa': kappa,
        'metric_scale': scale,
        'triples_evaluated': int(len(triples)),
        'graph_side_recomputation_matches_the_frozen_target': {
            'max_abs_difference': float(np.max(np.abs(graph_values - frozen))),
            'matches': bool(np.allclose(graph_values, frozen, atol=1e-9, rtol=1e-9)),
        },
        'hard_sid_decoded': {
            'sign_agreement': fidelity.sign_agreement(graph_values, quantized_values),
            'scale_factor': aligned,
            'aligned_mean_abs_error': float(np.abs(graph_values - aligned * quantized_values).mean()),
            'unaligned_mean_abs_error': float(np.abs(graph_values - quantized_values).mean()),
            'distribution': fidelity.distribution_difference(graph_values, quantized_values),
            'distance_distortion': fidelity.distance_distortion(
                np.concatenate(graph_distances), np.concatenate(quantized_distances)),
        },
        'continuous_z': {
            'note': 'reported separately; it never substitutes for the discrete SID',
            'sign_agreement': fidelity.sign_agreement(graph_values, continuous_values),
            'scale_factor': fidelity.scale_factor(graph_values, continuous_values),
        },
        'per_item_view': fidelity.per_item_distribution(graph_values, quantized_values,
                                                       triples[:, 0]),
    }
    return report


def per_item_view(graph_values: np.ndarray, quantized_values: np.ndarray,
                  triples: np.ndarray) -> dict:
    """The per-item view of the two curvature distributions."""
    return fidelity.per_item_distribution(graph_values, quantized_values, triples[:, 0])
