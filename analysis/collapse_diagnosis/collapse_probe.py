#!/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
STAGE2 = ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
sys.path.insert(0, str(STAGE2))
sys.path.insert(0, str(STAGE2 / "scripts"))
sys.path.insert(0, str(ROOT / "analysis/positive_negative_competition"))
import numpy as np
import torch
import train_rqvae as T
from model import RQVAE
import screen_competition as SCREEN
import competition as C

ARMS = ("resample", "E0", "H1", "B_E", "B_H")
OUT = ROOT / "results/collapse_diagnosis/collapse_probe.json"
PAIRS = SCREEN._load_pairs()


def config_for(arm):
    cfg = T._tokenizer_config()
    return cfg


def stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}


def auc_space(vectors):
    anchors = np.asarray(PAIRS["anchors"], dtype=np.int64).reshape(-1)
    positives = np.asarray(PAIRS["positives"], dtype=np.int64).reshape(-1, 1)
    negatives = np.asarray(PAIRS["negatives"], dtype=np.int64)
    pos, neg = C._distance_arrays(vectors, anchors, positives, negatives)
    return float(C.behavioral_auc(pos, neg))


def flat_grad(parameters):
    pieces = [parameter.grad.detach().reshape(-1) for parameter in parameters if parameter.grad is not None]
    return torch.cat(pieces) if pieces else torch.zeros(0)


def grad_compare(reconstruction, behaviour):
    rn = torch.linalg.vector_norm(reconstruction)
    bn = torch.linalg.vector_norm(behaviour)
    cosine = float(torch.dot(reconstruction, behaviour) / (rn * bn)) if rn.item() and bn.item() else float("nan")
    ratio = float(bn / rn) if rn.item() else float("inf")
    return {"cosine": cosine, "norm_ratio": ratio, "reconstruction_norm": float(rn), "behaviour_norm": float(bn)}


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUT}")
    embeddings_np = np.load(ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy").astype(np.float32, copy=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = torch.from_numpy(embeddings_np).to(device)
    all_anchor = np.asarray(PAIRS["anchors"], dtype=np.int64).reshape(-1)
    all_positive = np.asarray(PAIRS["positives"], dtype=np.int64).reshape(-1)
    all_negative = np.asarray(PAIRS["negatives"], dtype=np.int64)
    anchor = torch.as_tensor(all_anchor[:512], dtype=torch.long, device=device)
    positive = torch.as_tensor(all_positive[:512], dtype=torch.long, device=device)
    output = {"arms": {}, "fixed_batch": {"pair_file": "behavior_pairs_valid_s42.npz", "pair_rows": int(len(anchor))}}
    temp_dir = Path(tempfile.mkdtemp(prefix="collapse_probe_", dir="/tmp"))
    try:
        for arm in ARMS:
            checkpoint = ROOT / f"results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/{arm}/rqvae_best.pth"
            staged = temp_dir / f"{arm}_rqvae_best.pth"
            shutil.copy2(checkpoint, staged)
            model = RQVAE(config_for(arm), in_dim=768).to(device)
            try:
                state = torch.load(staged, map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(staged, map_location=device)
            model.load_state_dict(state.get("state_dict", state), strict=False)
            model.eval()
            raw = model
            behavior_items = embeddings[torch.cat((anchor, positive))]

            model.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _unused, _tokens = model(behavior_items)
            reconstruction_loss = raw.compute_loss(behavior_items, reconstructed, quant_loss)[0]
            reconstruction_loss.backward()
            encoder_parameters = tuple(raw.encoder.parameters())
            reconstruction_gradient = flat_grad(encoder_parameters)
            last_linear = [layer for layer in raw.encoder.modules() if isinstance(layer, torch.nn.Linear)][-1]
            reconstruction_last = flat_grad(tuple(last_linear.parameters()))

            model.zero_grad(set_to_none=True)
            behavior_residuals = T._residuals_with_trace(raw, behavior_items)
            unweighted_behavior = T.behavior_contrastive_loss(behavior_residuals, anchor, positive)
            behavior_weight = T.behavior_weight(T.EPOCHS)
            weighted_behavior = behavior_weight * unweighted_behavior
            weighted_behavior.backward()
            behavior_gradient = flat_grad(encoder_parameters)
            behavior_last = flat_grad(tuple(last_linear.parameters()))
            gradient = {"encoder": grad_compare(reconstruction_gradient, behavior_gradient),
                        "last_layer": grad_compare(reconstruction_last, behavior_last),
                        "behavior_weight": float(behavior_weight),
                        "temperature": float(T.BEHAVIOR_TEMPERATURE)}

            with torch.no_grad():
                continuous = raw.encoder(embeddings)
                residuals = T._residuals_with_trace(raw, embeddings)
                token_ids = raw.get_indices(embeddings)
                cumulative = torch.zeros_like(continuous)
                for level, layer in enumerate(raw.rq.vq_layers):
                    cumulative += layer.get_code_embs()[token_ids[:, level]]
                spaces = {"continuous": continuous.cpu().numpy()}
                for level in range(3):
                    spaces[f"residual_{level+1}"] = residuals[level].cpu().numpy()
                spaces["cumulative_quantised"] = cumulative.cpu().numpy()
                aucs = {name: auc_space(value) for name, value in spaces.items()}

                z = continuous.cpu().numpy()
                positive_distance, negative_distance = C._distance_arrays(z, all_anchor, all_positive[:, None], all_negative)
                distance = {"positive": stats(positive_distance), "negative": stats(negative_distance)}
                geometry = T._GEOMETRY_ARM_CONFIGS.get(arm, {})
                if geometry.get("mapping") == "expmap0":
                    norm = np.linalg.norm(z, axis=1)
                    mapped_radius = np.tanh(norm)
                    distance["mapped_radius"] = {
                        "p05": float(np.percentile(mapped_radius, 5)), "median": float(np.median(mapped_radius)),
                        "p95": float(np.percentile(mapped_radius, 95)),
                        "fraction_below_0_1": float(np.mean(mapped_radius < .1)),
                        "fraction_above_0_9": float(np.mean(mapped_radius > .9))}
                assignments = token_ids.cpu().numpy()
                level_health = []
                for level in range(3):
                    counts = np.bincount(assignments[:, level], minlength=256).astype(np.float64)
                    probabilities = counts[counts > 0] / len(assignments)
                    ordered_counts = np.sort(counts)
                    gini = float(2 * np.dot(np.arange(1, 257), ordered_counts) /
                                 (256 * counts.sum()) - 257 / 256)
                    level_health.append({
                        "level": level + 1, "distinct_codes": int(np.count_nonzero(counts)), "capacity": 256,
                        "top1_share": float(counts.max() / counts.sum()),
                        "normalized_entropy": float(-(probabilities * np.log(probabilities)).sum() / np.log(256)),
                        "gini": gini})
                _, joint_counts = np.unique(assignments, axis=0, return_counts=True)
                joint_probabilities = joint_counts / joint_counts.sum()
                joint_entropy = float(-(joint_probabilities * np.log(joint_probabilities)).sum())
                extended = T._extend_collisions(assignments, [256, 256, 256])
                code_health = {"levels": level_health,
                    "three_digit_uniqueness": float(len(joint_counts) / len(assignments)),
                    "full_sid_uniqueness": float(len(np.unique(extended, axis=0)) / len(extended)),
                    "joint_assignment_entropy_nats": joint_entropy}

            separations = {}
            flat_p = positive_distance.reshape(-1)
            flat_n = negative_distance.reshape(-1)
            separations["mean_negative_minus_positive"] = float(flat_n.mean() - flat_p.mean())
            separations["negative_not_farther_fraction"] = float(np.mean(flat_n <= flat_p[:, None]))
            output["arms"][arm] = {"gradient": gradient, "auc": aucs, "distance": distance,
                                   "separation": separations, "codebook_health": code_health}
            del model, state
    finally:
        shutil.rmtree(temp_dir)

    arms = output["arms"]
    print("SECTION A | arm | encoder_cosine | encoder_norm_ratio | last_layer_cosine | last_layer_norm_ratio")
    for arm in ARMS:
        g = arms[arm]["gradient"]
        print(f"{arm} | {g['encoder']['cosine']:.8g} | {g['encoder']['norm_ratio']:.8g} | {g['last_layer']['cosine']:.8g} | {g['last_layer']['norm_ratio']:.8g}")
    print("SECTION B | arm | level | continuous_auc | residual_auc | cumulative_quantised_auc")
    for arm in ARMS:
        metrics = arms[arm]["auc"]
        for level in (1, 2, 3):
            print(f"{arm} | L{level} | {metrics['continuous']:.8g} | {metrics[f'residual_{level}']:.8g} | {metrics['cumulative_quantised']:.8g}")
    print("SECTION C | arm | positive(mean,median,p05,p95) | negative(mean,median,p05,p95) | mapped_radius(p05,median,p95) | below_0.1 | above_0.9 | separation")
    for arm in ARMS:
        d = arms[arm]["distance"]
        p = d["positive"]; n = d["negative"]; r = d.get("mapped_radius", {})
        separation = arms[arm]["separation"]
        state = "separate" if p["mean"] < n["mean"] else "overlap"
        print(f"{arm} | {tuple(p[k] for k in ('mean','median','p05','p95'))} | {tuple(n[k] for k in ('mean','median','p05','p95'))} | {tuple(r.get(k) for k in ('p05','median','p95'))} | {r.get('fraction_below_0_1')} | {r.get('fraction_above_0_9')} | {state}, mean gap={separation['mean_negative_minus_positive']:.8g}, negative<=positive={separation['negative_not_farther_fraction']:.8g}")
    print("SECTION D | arm | level | distinct/256 | top1_share | normalized_entropy | gini | three_digit_uniqueness | full_sid_uniqueness | joint_assignment_entropy_nats")
    reference = arms["resample"]["codebook_health"]
    for arm in ARMS:
        ch = arms[arm]["codebook_health"]
        for level in range(3):
            h = ch["levels"][level]
            print(f"{arm} | L{level+1} | {h['distinct_codes']}/256 | {h['top1_share']:.8g} | {h['normalized_entropy']:.8g} | {h['gini']:.8g} | {ch['three_digit_uniqueness']:.8g} | {ch['full_sid_uniqueness']:.8g} | {ch['joint_assignment_entropy_nats']:.8g} | reference={reference['levels'][level]['distinct_codes']}/256,{reference['levels'][level]['top1_share']:.8g},{reference['levels'][level]['normalized_entropy']:.8g},{reference['levels'][level]['gini']:.8g}")
    print("Interpretation | Positive/negative co-clustering is indicated by distance overlap; gradient conflict by negative gradient cosine; codebook assignment degradation by reduced distinctness/entropy and uniqueness relative to resample.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
