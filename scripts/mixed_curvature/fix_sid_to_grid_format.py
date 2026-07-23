"""Task 450 P0.4 — Fix SID tensor to GRID's expected format.

GRID expects semantic_id tensor in (D, N) shape with D = num_hierarchies + 1
(the +1 is a dedup column from `deduplicate_rows_in_tensor`).

Our P0.2/P0.3 inference outputs `(N, 3)` float32 with codes L1/L2/L3 only.
Steps to convert:
  1. Load `(N, 3)` → `data`
  2. Apply `deduplicate_rows_in_tensor` logic → `(N, 4)` int64
  3. Transpose → `(4, N)` int64 (matches task15 reference exactly)

Verification against task15 reference:
  - Last col is dedup value (0=no collision, 1..N-1=position in collision group)
  - Number of unique dedup values should be small (task15 has 21)
  - dtype must be int64 (NOT float32 — GRID's id_map uses .long())

Why dtype matters: pre_processing.py:141 calls id_map.t()[v].view(-1)
where v is item index (long). If id_map is float32, embedding lookup may fail
or give wrong values.
"""
import sys
import torch
from pathlib import Path

GRID_ROOT = Path('/home/wlia0047/ar57/wenyu/GeneRec/GRID')


def add_dedup_column(data: torch.Tensor) -> torch.Tensor:
    """Mirrors GRID's `deduplicate_rows_in_tensor` (src/utils/tensor_utils.py:125)."""
    assert data.dim() == 2, f"expected 2D, got {data.shape}"
    unique_rows, inverse_indices, counts = torch.unique(
        data, dim=0, return_inverse=True, return_counts=True
    )
    output_indices = torch.zeros_like(inverse_indices)
    duplicate_indices = torch.where(counts > 1)[0]
    for i in range(len(duplicate_indices)):
        num_of_collisions = counts[duplicate_indices[i]]
        indices_to_change = torch.where(inverse_indices == duplicate_indices[i])[0]
        range_to_add = torch.arange(1, num_of_collisions + 1)
        output_indices = output_indices.scatter(0, indices_to_change, range_to_add)
    return torch.cat((data, output_indices.unsqueeze(1)), dim=1).long()


def fix_one(name: str, in_path: Path, out_path: Path) -> dict:
    data = torch.load(in_path, map_location='cpu', weights_only=False).long()
    print(f'[{name}] loaded {tuple(data.shape)} dtype={data.dtype}')
    if data.shape[1] == 3:
        # (N, 3) → add dedup → (N, 4) → transpose → (4, N)
        result_n4 = add_dedup_column(data)
        print(f'[{name}] after dedup: {tuple(result_n4.shape)} unique l1={len(torch.unique(result_n4[:,0]))} '
              f'l2={len(torch.unique(result_n4[:,1]))} l3={len(torch.unique(result_n4[:,2]))} '
              f'dedup_unique={len(torch.unique(result_n4[:,3]))} '
              f'dedup_min={result_n4[:,3].min().item()} dedup_max={result_n4[:,3].max().item()}')
        # (N, 4) → (4, N)
        sid = result_n4.t().contiguous()
    elif data.shape[0] == 3 and data.shape[1] == 4:
        # Already (4, N), just fix dtype if needed
        sid = data.long().contiguous()
        print(f'[{name}] already (4, N) format')
    elif data.shape[0] == 4:
        # (4, N) — just fix dtype
        sid = data.long().contiguous()
    else:
        raise ValueError(f'Unexpected shape {tuple(data.shape)} for {name}')
    print(f'[{name}] final SID: shape={tuple(sid.shape)} dtype={sid.dtype} '
          f'min={sid.min().item()} max={sid.max().item()}')
    torch.save(sid, out_path)
    print(f'[{name}] saved to {out_path}')
    return {
        'name': name,
        'in_shape': tuple(data.shape),
        'out_shape': tuple(sid.shape),
        'out_path': str(out_path),
    }


def main():
    targets = [
        ('P0.2', GRID_ROOT / 'logs/inference/runs/task58_p0_euclidean_concat_s22/pickle/merged_predictions_tensor.pt'),
        ('P0.3', GRID_ROOT / 'logs/inference/runs/task58_p0_mixed_curvature_s22/pickle/merged_predictions_tensor.pt'),
    ]
    summary = []
    for name, p in targets:
        if not p.exists():
            print(f'[{name}] NOT FOUND at {p}', file=sys.stderr)
            raise FileNotFoundError(p)
        info = fix_one(name, p, p)  # in-place overwrite
        summary.append(info)
    print('\n=== Summary ===')
    for s in summary:
        print(s)


if __name__ == '__main__':
    main()