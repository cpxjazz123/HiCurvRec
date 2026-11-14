"""Tree-Attention over the SID trie, and whether it recovers local behaviour structure.

The owner asked for the second stage of the method: instead of scoring pairs with a hand-written
combination of a distance and a tree relation, let a small attention model read the tree.

The model, per arm, with the quantized vectors and the SID table frozen:

    token(i)      Linear(quantized vector of i)
    attention     every item attends to a fixed set of partners -- the items in its level-2 branch
                  (siblings), a sample of the items in its level-1 branch (cousins) and itself --
                  with a learned bias per relation type (self / sibling / cousin)
    h_i           the attended representation
    scorer        the same small pair model as the relation experiment: [u-v, u*v] -> probability

Three variants, same architecture, same parameters, same data split:

    tree       the real SID trie
    shuffled   the trie with the items' positions permuted: the same code histogram, no structure
    random     attention over partners drawn at random: no tree at all, same parameter count

The evaluation is the owner's: for every structure instance of the real behaviour graph, is the
real edge scored above the non-edges that share an endpoint with it, and above the one-step-away
negatives.

    python3 results/research/issue215/tree_attention_recoverability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
VECTORS = OUT / "vectors"
GRAPHS = OUT / "graphs"
JOURNAL = OUT / "tree_attention_recoverability.json"
#: which arms to run; overridable so an arm can be added without touching the rest
ARMS = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("F0", "C")
SEED = 42
EPOCHS = 600
HIDDEN = 64
HEADS = 4
PARTNERS = 16          # siblings and cousins each
LR = 2e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(OUT))
from sid_tree_recoverability import SID_PATHS, load_sid, shuffled_sid  # noqa: E402


def groups_from_sid(sid: dict[int, tuple], keys: list[tuple[int, int]]) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for item in keys:
        code = sid[item]
        key = tuple(code[:depth] for depth in (1, 2))
        groups.setdefault(key, []).append(item)
    return groups


def partner_index(sid: dict[int, tuple], items: list[int], rng: np.random.Generator) -> dict[int, tuple[list[int], list[int], list[int]]]:
    """For every item: its level-2 siblings, a sample of its level-1 cousins, and itself."""
    level1: dict[int, list[int]] = {}
    level2: dict[tuple[int, int], list[int]] = {}
    for item in items:
        code = sid[item]
        level1.setdefault(code[0], []).append(item)
        level2.setdefault((code[0], code[1]), []).append(item)
    out = {}
    for item in items:
        code = sid[item]
        siblings = [x for x in level2[(code[0], code[1])] if x != item]
        cousins_pool = [x for x in level1[code[0]] if x not in level2[(code[0], code[1])]]
        cousins = [int(x) for x in rng.choice(len(cousins_pool), size=min(PARTNERS, len(cousins_pool)),
                                           replace=False)] if cousins_pool else []
        cousins = [cousins_pool[i] for i in cousins]
        out[item] = (siblings[:PARTNERS], cousins, [item])
    return out


class TreeAttention(torch.nn.Module):
    """One attention layer over the partners, with a learned bias per relation type."""

    def __init__(self, dimension: int, heads: int):
        super().__init__()
        self.heads = heads
        self.depth = dimension // heads
        self.query = torch.nn.Linear(dimension, dimension)
        self.key = torch.nn.Linear(dimension, dimension)
        self.value = torch.nn.Linear(dimension, dimension)
        self.bias = torch.nn.Parameter(torch.zeros(3))          # self / sibling / cousin
        self.output = torch.nn.Linear(dimension, dimension)

    def forward(self, tokens: torch.Tensor, partners: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        """tokens [n, d]; partners [n, p, d]; kinds [n, p] in {0 self, 1 sibling, 2 cousin}."""
        batch, count, _ = partners.shape
        query = self.query(tokens).view(batch, self.heads, 1, self.depth)
        key = self.key(partners).view(batch, count, self.heads, self.depth).permute(0, 2, 3, 1)
        value = self.value(partners).view(batch, count, self.heads, self.depth).permute(0, 2, 1, 3)
        score = (query @ key) / np.sqrt(self.depth)                 # [batch, heads, 1, count]
        score = score + self.bias[kinds][:, None, None, :]         # one learned bias per relation
        weight = torch.softmax(score, dim=-1)
        context = (weight @ value).reshape(batch, self.heads * self.depth)
        return tokens + self.output(context)


class Scorer(torch.nn.Module):
    """The same pair model as the relation experiment: [u-v, u*v] -> a probability."""

    def __init__(self, dimension: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2 * dimension, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([left - right, left * right], dim=1)).ravel()


def build_pairs(edges: np.ndarray, negatives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_pairs = edges
    false_pairs = negatives
    return true_pairs, false_pairs


def train_variant(arm: str, variant: str, items: list[int], sid: dict[int, tuple],
                  vectors: np.ndarray, edges: np.ndarray, negatives: np.ndarray,
                  train_true: np.ndarray, train_false: np.ndarray,
                  partners: dict | None = None) -> tuple[Scorer, torch.Tensor]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if partners is None:
        partners = partner_index(sid, items, rng)
    width = max(len(value[0]) + len(value[1]) + 1 for value in partners.values())
    index = {item: position for position, item in enumerate(items)}
    token = torch.nn.Linear(vectors.shape[1], HIDDEN).to(DEVICE)
    encoder = TreeAttention(HIDDEN, HEADS).to(DEVICE)
    scorer = Scorer(HIDDEN).to(DEVICE)
    optimiser = torch.optim.Adam(list(token.parameters()) + list(encoder.parameters())
                                 + list(scorer.parameters()), lr=LR)

    base = torch.tensor(vectors, dtype=torch.float32, device=DEVICE)
    neighbour_rows = torch.zeros(len(items), width, dtype=torch.long, device=DEVICE)
    kind_rows = torch.zeros(len(items), width, dtype=torch.long, device=DEVICE)
    mask_rows = torch.zeros(len(items), width, dtype=torch.bool, device=DEVICE)
    for item in items:
        row = index[item]
        siblings, cousins, self_list = partners[item]
        cursor = 0
        for kind, group in ((1, siblings), (2, cousins), (0, self_list)):
            for partner in group:
                if cursor >= width:
                    break
                neighbour_rows[row, cursor] = index[partner]
                kind_rows[row, cursor] = kind
                mask_rows[row, cursor] = True
                cursor += 1

    def representations() -> torch.Tensor:
        hidden = token(base)
        # the partners are read through the same projection, so the attention sees tokens
        gathered = hidden[neighbour_rows]
        hidden = encoder(hidden, gathered, kind_rows)
        return hidden

    true_tensor = torch.tensor(edges, dtype=torch.long, device=DEVICE)
    false_tensor = torch.tensor(negatives, dtype=torch.long, device=DEVICE)
    train_true = torch.tensor(train_true, dtype=torch.long, device=DEVICE)
    train_false = torch.tensor(train_false, dtype=torch.long, device=DEVICE)
    for _ in range(EPOCHS):
        hidden = representations()
        logits = scorer(hidden[true_tensor[train_true][:, 0]], hidden[true_tensor[train_true][:, 1]])
        negatives_logits = scorer(hidden[false_tensor[train_false][:, 0]],
                                 hidden[false_tensor[train_false][:, 1]])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits)) + torch.nn.functional.binary_cross_entropy_with_logits(
            negatives_logits, torch.zeros_like(negatives_logits))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return scorer, representations()


def score_pairs(scorer: Scorer, hidden: torch.Tensor, pairs: np.ndarray) -> np.ndarray:
    tensor = torch.tensor(pairs, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return scorer(hidden[tensor[:, 0]], hidden[tensor[:, 1]]).cpu().numpy()


def main() -> int:
    from relation_scores import local_negatives
    from relative_distance_structures import build_instances, load_behaviour

    n, edges, adjacency = load_behaviour()
    degrees = np.asarray([len(adjacency[i]) for i in range(n)], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    negatives = local_negatives(edges, adjacency, n, rng)
    instances = build_instances(n, edges, adjacency, degrees, np.random.default_rng(SEED))

    comparisons: dict[str, list[tuple]] = {}
    for name in ("path3", "path4", "star4", "cycle4", "triangle_tail", "diamond"):
        pairs = []
        for instance in instances[name]:
            for true_pair in list(instance["true"]) + list(instance["inner"]):
                for false_pair in instance["false"]:
                    if set(true_pair) == set(false_pair):
                        continue
                    if not (set(true_pair) & set(false_pair)):
                        continue
                    pairs.append((tuple(true_pair), tuple(false_pair)))
        comparisons[name] = pairs

    # external: the true edge against the one-step-away negatives
    external = []
    for a, b in edges.tolist():
        a, b = int(a), int(b)
        for item, other in ((a, b), (b, a)):
            for w in list(adjacency[other])[:4]:
                if w == item or w in adjacency[item]:
                    continue
                external.append(((item, other), (min(item, w), max(item, w))))
    comparisons["edge (hard negatives)"] = external

    order = rng.permutation(len(edges))
    cut = int(0.8 * len(edges))
    report = {"structures": {}}
    for arm in ARMS:
        vectors = np.load(VECTORS / f"{arm}_q.npy").astype(np.float32)
        sid = load_sid(arm)
        items = sorted(sid)
        negative_order = rng.permutation(len(negatives))
        negative_cut = int(0.8 * len(negatives))
        block = {}
        for variant in ("tree", "shuffled", "random"):
            rng_variant = np.random.default_rng(SEED + 11)
            variant_sid = shuffled_sid(sid, rng_variant) if variant == "shuffled" else sid
            tree_partners = partner_index(variant_sid, items, rng_variant)
            if variant == "random":
                # same architecture, no tree: the partners are drawn at random
                partners = {}
                for item in items:
                    siblings, cousins, self_list = tree_partners[item]
                    partners[item] = ([int(rng_variant.integers(len(items))) for _ in range(len(siblings))],
                                       [int(rng_variant.integers(len(items))) for _ in range(len(cousins))],
                                       self_list)
            else:
                partners = tree_partners
            scorer, hidden = train_variant(arm, variant, items, variant_sid, vectors, edges, negatives,
                                           order[:cut], negative_order[:negative_cut], partners=partners)
            scores = {}
            for name, pairs in comparisons.items():
                true_pairs = [pair[0] for pair in pairs]
                false_pairs = [pair[1] for pair in pairs]
                true_score = score_pairs(scorer, hidden, true_pairs)
                false_score = score_pairs(scorer, hidden, false_pairs)
                wins = float(np.mean(np.where(true_score > false_score, 1.0,
                                             np.where(true_score == false_score, 0.5, 0.0))))
                scores[name] = {"score": wins, "comparisons": len(pairs)}
            block[variant] = scores
            report["structures"][arm] = block
            JOURNAL.write_text(json.dumps(report, indent=1, default=float))
            print(f"{arm:<4} {variant:<9} " + "  ".join(f"{name}: {value['score']:.4f}"
                                                     for name, value in scores.items()), flush=True)
        report["structures"][arm] = block

    JOURNAL.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwritten: {JOURNAL.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
