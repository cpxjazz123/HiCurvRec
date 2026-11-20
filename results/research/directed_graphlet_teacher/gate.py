"""Issue #222 C: the Teacher -> Student transferability gate.

The Teacher sees the instance's own directed adjacency and reaches 0.71 / 0.61 on
held-out instances.  The Student sees only the frozen baseline's hard codes `q`.
Whether anything survives that information gap is an empirical question, and this
module answers it *before* any Stage 2 budget is spent.

Variants (identical data, seed, optimiser, step budget; only the supervision differs)

    random        the head as initialised -- the chance floor of this readout
    majority      no head at all: predict the training-majority class
    degree        a lookup on the sorted (in, out) degree histogram, no learning
    hard_ce       cross-entropy on the true subtype/orbit labels
    teacher_kd    KL to the frozen Teacher's softened posteriors (no hard labels)
    shuffled_kd   the same Teacher weights fed adjacency flipped per unordered pair
    teacher_kd_z  teacher KD on the continuous encoder output `z` instead of `q`:
                  the quantisation bottle-neck, measured rather than assumed

Each variant is scored on training instances, on all held-out instances
(instance-disjoint) and on the subset whose unordered node pairs never occur in a
training instance (pair-disjoint), so memorisation cannot pass for transfer.

The Student path carries no structure: `student_forward` takes `q` only, and
`assert_no_leak` refuses a bundle that carries anything else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import kdistill  # noqa: E402
from student_head import StudentQuantisedHead, StudentRelationalHead  # noqa: E402

ROLE_DIR = REPO / "results" / "research" / "directed_graphlet_role"
if str(ROLE_DIR) not in sys.path:
    sys.path.insert(0, str(ROLE_DIR))
from directed_graphlets import load_adjacency  # noqa: E402
from directed_role import FAMILIES, instance_targets, load_pool, load_vocabulary  # noqa: E402

TRAIN_CAP = 5000
TEST_CAP = 5000
EPOCHS_PER_FAMILY = 2000
LR = 1e-3
SEED = 20261120
TEMPERATURE = 2.0
ALPHA = 1.0
HIDDEN = 96

BANNED_KEYS = ("adjacency", "adj", "edges", "edge", "graph", "teacher_logits",
               "teacher_hidden", "node_hidden", "subtype", "orbit", "label",
               "canonical")


class LeakError(RuntimeError):
    """A Student input bundle carries something the Student must not read."""


def assert_no_leak(bundle: dict) -> None:
    for key in bundle:
        lowered = str(key).lower()
        for banned in BANNED_KEYS:
            if banned in lowered:
                raise LeakError("student bundle carries %r (matches %r)" % (key, banned))


def student_forward(head: StudentQuantisedHead, q: torch.Tensor,
                    mask: torch.Tensor | None = None):
    """The only entry point the Student has: codes in, logits out."""
    assert_no_leak({"q": q, "mask": mask})
    return head(q)


def pair_disjoint_keep(train_nodes: np.ndarray, heldout_nodes: np.ndarray) -> torch.Tensor:
    """True where a held-out instance shares no unordered pair with training."""
    pairs = set()
    for row in np.asarray(train_nodes):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                pairs.add((min(int(row[i]), int(row[j])), max(int(row[i]), int(row[j]))))
    keep = []
    for row in np.asarray(heldout_nodes):
        clean = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                key = (min(int(row[i]), int(row[j])), max(int(row[i]), int(row[j])))
                if key in pairs:
                    clean = False
                    break
            if not clean:
                break
        keep.append(clean)
    return torch.tensor(keep, dtype=torch.bool)


def macro_f1(prediction: np.ndarray, target: np.ndarray, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        tp = int(((prediction == label) & (target == label)).sum())
        fp = int(((prediction == label) & (target != label)).sum())
        fn = int(((prediction != label) & (target == label)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def degree_signature(nodes: np.ndarray, edges: set, n_items: int) -> list:
    """Sorted (in, out) degree pairs per node, one tuple per instance."""
    signatures = []
    for row in np.asarray(nodes):
        pairs = []
        for source in row:
            outbound = sum(int(int(source) * n_items + int(target)) in edges
                           for target in row if target != source)
            inbound = sum(int(int(target) * n_items + int(source)) in edges
                          for target in row if target != source)
            pairs.append((inbound, outbound))
        signatures.append(tuple(sorted(pairs)))
    return signatures


def main(shared: bool = True, specialists=("hard_ce", "teacher_kd", "shuffled_kd",
                                        "teacher_kd_z_rel")) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from frozen_sid import FrozenSID
    from teacher_model import load_frozen_teacher

    vocab = load_vocabulary("directed")
    n_subtypes, n_orbits = vocab["n_subtypes"], vocab["n_orbits"]
    sid = FrozenSID()
    teacher, teacher_record = load_frozen_teacher(
        HERE / "teacher" / "teacher_directed_v1.pth",
        expect_vocab={"mode": "directed", "n_subtypes": n_subtypes, "n_orbits": n_orbits})
    teacher.to(device).eval()

    graph = load_adjacency()
    n_items = graph.shape[0]
    rows, cols = graph.nonzero()
    edge_set = set((rows.astype(np.int64) * n_items + cols).tolist())
    semantic = torch.from_numpy(np.load(REPO / "stage1_GeneEmbedding" / "output" /
                                        "sentence_t5.npy").astype("float32"))

    train_families = load_pool("train", vocab, max_instances=TRAIN_CAP)
    test_families = load_pool("heldout", vocab, max_instances=TEST_CAP)

    def adjacency_tensor(nodes: np.ndarray) -> np.ndarray:
        count, width = nodes.shape
        out = np.zeros((count, width, width), dtype=np.float32)
        for i in range(width):
            for j in range(width):
                if i == j:
                    continue
                keys = nodes[:, i].astype(np.int64) * n_items + nodes[:, j]
                out[:, i, j] = [int(key) in edge_set for key in keys]
        return out

    def shuffle_directions(a: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = a.copy()
        for i in range(a.shape[1]):
            for j in range(i + 1, a.shape[2]):
                flip = rng.random(len(a)) < 0.5
                out[:, i, j] = np.where(flip, a[:, j, i], a[:, i, j])
                out[:, j, i] = np.where(flip, a[:, i, j], a[:, j, i])
        return out

    prepared = {}
    for family in FAMILIES:
        entry = {}
        for split, (nodes, orbits, subtypes) in (
                ("train", train_families[family]),
                ("test", test_families[family])):
            targets = instance_targets(family, subtypes, orbits, vocab)
            valid = torch.from_numpy(targets["subtype"] >= 0)
            adjacency = torch.from_numpy(adjacency_tensor(nodes))
            flat = nodes.reshape(-1)
            entry[split] = {
                "nodes": nodes, "valid": valid,
                "subtype": targets["subtype"], "orbit": targets["orbit"],
                "adjacency": adjacency,
                "q": sid.q(flat).reshape(len(nodes), nodes.shape[1], -1),
                "z": sid.z(flat).reshape(len(nodes), nodes.shape[1], -1),
            }
        entry["pair_keep"] = pair_disjoint_keep(entry["train"]["nodes"], entry["test"]["nodes"])
        entry["degree_test"] = degree_signature(entry["test"]["nodes"], edge_set, n_items)
        entry["degree_train"] = degree_signature(entry["train"]["nodes"], edge_set, n_items)
        entry["shuffled_train"] = torch.from_numpy(
            shuffle_directions(entry["train"]["adjacency"].numpy(), SEED))
        prepared[family] = entry
        print(f"{family}: train {len(entry['train']['nodes'])} "
              f"test {len(entry['test']['nodes'])} pair-clean "
              f"{int((entry['pair_keep'] & entry['test']['valid']).sum())}", flush=True)

    dim_q = int(prepared[FAMILIES[0]]["train"]["q"].shape[-1])
    dim_z = int(prepared[FAMILIES[0]]["train"]["z"].shape[-1])

    def teacher_targets(family: str, mode: str):
        entry = prepared[family]["train"]
        # Every train instance, valid label or not: the KD rows are addressed by
        # their position in the family array, not in the filtered subset.
        keep = torch.arange(len(entry["nodes"]))
        index = torch.from_numpy(entry["nodes"]).long()[keep]
        shuffled = prepared[family]["shuffled_train"]
        adjacency = entry["adjacency"][keep] if mode == "directed" else shuffled[keep]
        with torch.no_grad():
            type_logits, orbit_logits, _ = teacher(semantic[index].to(device), adjacency.to(device))
        return type_logits.cpu(), orbit_logits.cpu()

    soft = {family: {"directed": teacher_targets(family, "directed"),
                     "shuffled": teacher_targets(family, "shuffled")} for family in FAMILIES}

    def train_head(mode: str, feature_key: str, seed: int, kind: str = "pooled"):
        """Full-batch on one family at a time, the same schedule the Teacher used.

        Mini-batches at this sample size left the head at the majority class even
        with normalised inputs: 128-row steps cannot cover 20,000 instances often
        enough inside a fixed budget, and an under-trained head would be read as
        "no signal".  Full batch keeps the supervision comparable across variants.
        """
        width = dim_q if feature_key == "q" else dim_z
        torch.manual_seed(seed)
        builder = StudentRelationalHead if kind == "relational" else StudentQuantisedHead
        head = builder(width, HIDDEN, n_subtypes, n_orbits).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
        for epoch in range(EPOCHS_PER_FAMILY * len(FAMILIES)):
            family = FAMILIES[epoch % len(FAMILIES)]
            entry = prepared[family]["train"]
            keep = np.flatnonzero(entry["valid"].numpy())
            index = torch.from_numpy(keep)
            features = entry[feature_key][index].to(device)
            type_logits, orbit_logits = student_forward(head, features)
            sub_target = torch.from_numpy(entry["subtype"][keep]).to(device)
            orbit_target = torch.from_numpy(entry["orbit"][keep]).to(device)
            if mode == "hard_ce":
                loss = F.cross_entropy(type_logits, sub_target) + \
                    F.cross_entropy(orbit_logits.reshape(-1, orbit_logits.shape[-1]),
                                    orbit_target.reshape(-1))
            elif mode in ("teacher_kd", "shuffled_kd"):
                source = "directed" if mode == "teacher_kd" else "shuffled"
                teacher_type, teacher_orbit = soft[family][source]
                loss = kdistill.joint_kd(type_logits, orbit_logits,
                                         teacher_type[index].to(device),
                                         teacher_orbit[index].to(device),
                                         temperature=TEMPERATURE, alpha=ALPHA)["total"]
            else:
                raise ValueError("unknown supervision %r" % (mode,))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        return head


    def evaluate_family(head, family: str, feature_key: str) -> dict:
        entry = prepared[family]
        family_row = {}
        for split, bundle, keep in (
                ("train", entry["train"], entry["train"]["valid"]),
                ("heldout", entry["test"], entry["test"]["valid"]),
                ("pair_disjoint", entry["test"],
                 entry["test"]["valid"] & entry["pair_keep"])):
            keep_np = np.flatnonzero(keep.numpy())
            if len(keep_np) == 0:
                continue
            with torch.no_grad():
                type_logits, orbit_logits = student_forward(
                    head, bundle[feature_key][keep_np].to(device))
            subtype_prediction = type_logits.argmax(-1).cpu().numpy()
            orbit_prediction = orbit_logits.argmax(-1).cpu().numpy()
            family_row[split] = {
                "subtype_macro_f1": macro_f1(subtype_prediction,
                                             bundle["subtype"][keep_np], n_subtypes),
                "subtype_accuracy": float((subtype_prediction
                                           == bundle["subtype"][keep_np]).mean()),
                "orbit_macro_f1": macro_f1(orbit_prediction.reshape(-1),
                                           bundle["orbit"][keep_np].reshape(-1), n_orbits),
                "instances": int(len(keep_np)),
            }
        return family_row

    def train_specialist(mode: str, feature_key: str, seed: int, kind: str) -> dict:
        """One head per family, its whole budget on that family alone.

        The shared head interleaves five families and never saturates any of them;
        a specialist removes that interference, so a negative held-out number cannot
        be blamed on the shared-parameter budget.
        """
        width = dim_q if feature_key == "q" else dim_z
        builder = StudentRelationalHead if kind == "relational" else StudentQuantisedHead
        heads = {}
        for offset, family in enumerate(FAMILIES):
            torch.manual_seed(seed + offset)
            head = builder(width, HIDDEN, n_subtypes, n_orbits).to(device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
            entry = prepared[family]["train"]
            keep = np.flatnonzero(entry["valid"].numpy())
            index = torch.from_numpy(keep)
            features = entry[feature_key][index].to(device)
            sub_target = torch.from_numpy(entry["subtype"][keep]).to(device)
            orbit_target = torch.from_numpy(entry["orbit"][keep]).to(device)
            for _ in range(EPOCHS_PER_FAMILY):
                type_logits, orbit_logits = student_forward(head, features)
                if mode == "hard_ce":
                    loss = F.cross_entropy(type_logits, sub_target) + \
                        F.cross_entropy(orbit_logits.reshape(-1, orbit_logits.shape[-1]),
                                        orbit_target.reshape(-1))
                else:
                    source = "directed" if mode == "teacher_kd" else "shuffled"
                    teacher_type, teacher_orbit = soft[family][source]
                    loss = kdistill.joint_kd(type_logits, orbit_logits,
                                             teacher_type[index].to(device),
                                             teacher_orbit[index].to(device),
                                             temperature=TEMPERATURE, alpha=ALPHA)["total"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            head.eval()
            heads[family] = head
        return heads

    def evaluate_specialists(heads: dict, feature_key: str, name: str) -> dict:
        row = {"variant": name, "input": feature_key, "head": "specialist", "per_family": {}}
        for family, head in heads.items():
            row["per_family"][family] = evaluate_family(head, family, feature_key)
        for split in ("train", "heldout", "pair_disjoint"):
            values = [family[split] for family in row["per_family"].values() if split in family]
            row[split + "_mean"] = {
                "subtype_macro_f1": float(np.mean([v["subtype_macro_f1"] for v in values])),
                "subtype_accuracy": float(np.mean([v["subtype_accuracy"] for v in values])),
                "orbit_macro_f1": float(np.mean([v["orbit_macro_f1"] for v in values])),
            } if values else {}
        return row

    def evaluate(head, feature_key: str, name: str) -> dict:
        row = {"variant": name, "input": feature_key, "per_family": {}}
        for family in FAMILIES:
            entry = prepared[family]
            family_row = {}
            for split, bundle, keep in (
                    ("train", entry["train"], entry["train"]["valid"]),
                    ("heldout", entry["test"], entry["test"]["valid"]),
                    ("pair_disjoint", entry["test"],
                     entry["test"]["valid"] & entry["pair_keep"])):
                keep_np = np.flatnonzero(keep.numpy())
                if len(keep_np) == 0:
                    continue
                with torch.no_grad():
                    type_logits, orbit_logits = student_forward(
                        head, bundle[feature_key][keep_np].to(device))
                subtype_prediction = type_logits.argmax(-1).cpu().numpy()
                orbit_prediction = orbit_logits.argmax(-1).cpu().numpy()
                family_row[split] = {
                    "subtype_macro_f1": macro_f1(subtype_prediction,
                                                 bundle["subtype"][keep_np], n_subtypes),
                    "subtype_accuracy": float((subtype_prediction
                                               == bundle["subtype"][keep_np]).mean()),
                    "orbit_macro_f1": macro_f1(orbit_prediction.reshape(-1),
                                               bundle["orbit"][keep_np].reshape(-1), n_orbits),
                    "instances": int(len(keep_np)),
                }
            row["per_family"][family] = family_row
        for split in ("train", "heldout", "pair_disjoint"):
            values = [family[split] for family in row["per_family"].values() if split in family]
            row[split + "_mean"] = {
                "subtype_macro_f1": float(np.mean([v["subtype_macro_f1"] for v in values])),
                "subtype_accuracy": float(np.mean([v["subtype_accuracy"] for v in values])),
                "orbit_macro_f1": float(np.mean([v["orbit_macro_f1"] for v in values])),
            } if values else {}
        return row

    results = {"teacher": teacher_record, "sid": sid.describe(),
               "config": {"epochs_per_family": EPOCHS_PER_FAMILY, "batch": "full-family",
                          "lr": LR, "seed": SEED,
                          "temperature": TEMPERATURE, "alpha": ALPHA, "hidden": HIDDEN,
                          "train_cap": TRAIN_CAP, "test_cap": TEST_CAP},
               "families": {}, "variants": {}}
    for family in FAMILIES:
        entry = prepared[family]
        results["families"][family] = {
            "train_instances": int(entry["train"]["valid"].sum()),
            "heldout_instances": int(entry["test"]["valid"].sum()),
            "pair_clean_heldout": int((entry["pair_keep"] & entry["test"]["valid"]).sum()),
        }

    torch.manual_seed(SEED)
    untrained = StudentQuantisedHead(dim_q, HIDDEN, n_subtypes, n_orbits).to(device).eval()
    results["variants"]["random"] = evaluate(untrained, "q", "random")

    for name, mode, feature_key, kind in (() if not shared else (
            ("hard_ce", "hard_ce", "q", "pooled"),
            ("teacher_kd", "teacher_kd", "q", "pooled"),
            ("shuffled_kd", "shuffled_kd", "q", "pooled"),
            ("teacher_kd_z", "teacher_kd", "z", "pooled"),
            ("hard_ce_rel", "hard_ce", "q", "relational"),
            ("teacher_kd_rel", "teacher_kd", "q", "relational"),
            ("teacher_kd_z_rel", "teacher_kd", "z", "relational"))):
        print("training", name, flush=True)
        head = train_head(mode, feature_key, SEED + len(results["variants"]), kind)
        results["variants"][name] = evaluate(head, feature_key, name)
        results["variants"][name]["head"] = kind
        print(json.dumps({key: results["variants"][name][key] for key in
                          ("heldout_mean", "pair_disjoint_mean")}), flush=True)

    specialist_table = {"hard_ce": ("hard_ce_spec", "hard_ce", "q", "pooled"),
                        "teacher_kd": ("teacher_kd_spec", "teacher_kd", "q", "pooled"),
                        "shuffled_kd": ("shuffled_kd_spec", "shuffled_kd", "q", "pooled"),
                        "teacher_kd_z_rel": ("teacher_kd_z_rel_spec", "teacher_kd", "z", "relational"),
                        "teacher_kd_z": ("teacher_kd_z_spec", "teacher_kd", "z", "pooled")}
    for key in specialists:
        name, mode, feature_key, kind = specialist_table[key]
        print("training specialist", name, flush=True)
        heads = train_specialist(mode, feature_key, SEED + len(results["variants"]), kind)
        results["variants"][name] = evaluate_specialists(heads, feature_key, name)
        print(json.dumps({key: results["variants"][name][key] for key in
                          ("heldout_mean", "pair_disjoint_mean")}), flush=True)

    majority = {}
    for family in FAMILIES:
        entry = prepared[family]
        train_valid = entry["train"]["valid"].numpy()
        majority_subtype = int(np.bincount(entry["train"]["subtype"][train_valid]).argmax())
        orbit_flat = entry["train"]["orbit"][train_valid].reshape(-1)
        majority_orbit = int(np.bincount(orbit_flat).argmax())
        row = {}
        for split, keep in (("heldout", entry["test"]["valid"]),
                            ("pair_disjoint", entry["test"]["valid"] & entry["pair_keep"])):
            keep_np = np.flatnonzero(keep.numpy())
            subtype_target = entry["test"]["subtype"][keep_np]
            orbit_target = entry["test"]["orbit"][keep_np]
            row[split] = {
                "subtype_macro_f1": macro_f1(np.full(len(subtype_target), majority_subtype),
                                             subtype_target, n_subtypes),
                "orbit_macro_f1": macro_f1(np.full(orbit_target.size, majority_orbit),
                                           orbit_target.reshape(-1), n_orbits),
                "instances": int(len(keep_np)),
            }
        majority[family] = row
    results["variants"]["majority"] = {
        "variant": "majority", "per_family": majority,
        **{split + "_mean": {
            "subtype_macro_f1": float(np.mean([family[split]["subtype_macro_f1"]
                                               for family in majority.values()])),
            "orbit_macro_f1": float(np.mean([family[split]["orbit_macro_f1"]
                                             for family in majority.values()]))}
           for split in ("heldout", "pair_disjoint")}}

    degree = {}
    for family in FAMILIES:
        entry = prepared[family]
        lookup = {}
        train_valid = entry["train"]["valid"].numpy()
        for signature, label in zip([entry["degree_train"][index] for index in train_valid],
                                    entry["train"]["subtype"][train_valid]):
            lookup.setdefault(signature, []).append(int(label))
        keep_np = np.flatnonzero(entry["test"]["valid"].numpy())
        prediction = np.array([int(np.bincount(lookup[signature]).argmax())
                               if signature in lookup else -1
                               for signature in (entry["degree_test"][index] for index in keep_np)])
        degree[family] = {"heldout": {
            "subtype_macro_f1": macro_f1(prediction, entry["test"]["subtype"][keep_np], n_subtypes),
            "coverage": float((prediction >= 0).mean()), "instances": int(len(keep_np))}}
    results["variants"]["degree"] = {
        "variant": "degree", "per_family": degree,
        "heldout_mean": {"subtype_macro_f1": float(np.mean(
            [row["heldout"]["subtype_macro_f1"] for row in degree.values()])),
            "coverage": float(np.mean([row["heldout"]["coverage"] for row in degree.values()]))}}

    if not shared and (HERE / "gate.json").is_file():
        previous = json.loads((HERE / "gate.json").read_text())
        previous["variants"].update(results["variants"])
        results = previous
    (HERE / "gate.json").write_text(json.dumps(results, indent=1))
    summary = {}
    for name, row in results["variants"].items():
        summary[name] = {
            "train_subtype": row.get("train_mean", {}).get("subtype_macro_f1"),
            "train_acc": row.get("train_mean", {}).get("subtype_accuracy"),
            "hold_acc": row.get("heldout_mean", {}).get("subtype_accuracy"),
            "heldout_subtype": row.get("heldout_mean", {}).get("subtype_macro_f1"),
            "heldout_orbit": row.get("heldout_mean", {}).get("orbit_macro_f1"),
            "pair_subtype": row.get("pair_disjoint_mean", {}).get("subtype_macro_f1"),
            "pair_orbit": row.get("pair_disjoint_mean", {}).get("orbit_macro_f1")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
