"""
Task 64 — Exp1 Step 2: 训练几何标签预测分类器

验证从原始 embedding 能否预测局部几何类型。

用法: python scripts/task6_exp1_train_classifier.py [--device cuda:0]
"""

import os, sys, argparse, json, time
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GeometryPredictor(nn.Module):
    """Simple MLP: 768-dim → 128 hidden → 3-dim softmax"""
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        return self.net(x)  # raw logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--labels", default="products/task16/from_task64/exp1_geo_labels.npy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("=" * 60)
    print("Task 64 — Exp1 Step 2: 几何标签预测分类器")
    print("=" * 60)

    # ── 加载数据 ──
    print(f"\n[1/4] Loading data...")
    embeddings = torch.load(args.embeddings, map_location="cpu").float()
    geo_probs = np.load(args.labels)  # (N, 3) softmax probabilities
    N, D = embeddings.shape
    print(f"  Embeddings: {list(embeddings.shape)}")
    print(f"  Geo labels: {geo_probs.shape}")

    # Hard labels for accuracy
    hard_labels = np.argmax(geo_probs, axis=1)

    # Train/test split
    X = embeddings.numpy()
    y_hard = hard_labels
    y_soft = geo_probs

    X_train, X_test, yh_train, yh_test, ys_train, ys_test = train_test_split(
        X, y_hard, y_soft, test_size=0.2, random_state=42, stratify=y_hard
    )

    X_train_t = torch.from_numpy(X_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    ys_train_t = torch.from_numpy(ys_train).float().to(device)
    yh_train_t = torch.from_numpy(yh_train).long().to(device)
    ys_test_t = torch.from_numpy(ys_test).float().to(device)
    yh_test_t = torch.from_numpy(yh_test).long().to(device)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 训练 ──
    print(f"\n[2/4] Training classifier ({args.epochs} epochs)...")
    model = GeometryPredictor(input_dim=D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Dual loss: cross-entropy (hard) + KL divergence (soft)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    history = {"train_acc": [], "test_acc": [], "train_kl": [], "test_kl": [], "train_cos": [], "test_cos": []}
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        # Hard loss
        loss_hard = ce_loss(logits, yh_train_t)
        # Soft loss (KL)
        pred_soft = torch.log_softmax(logits, dim=1)
        loss_soft = kl_loss(pred_soft, ys_train_t)

        loss = loss_hard + 0.5 * loss_soft
        loss.backward()
        optimizer.step()

        # Eval
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Train metrics
                train_logits = model(X_train_t)
                train_pred = train_logits.argmax(dim=1)
                train_acc = (train_pred == yh_train_t).float().mean().item()

                train_soft = torch.softmax(train_logits, dim=1)
                train_cos = nn.functional.cosine_similarity(train_soft, ys_train_t).mean().item()

                # Test metrics
                test_logits = model(X_test_t)
                test_pred = test_logits.argmax(dim=1)
                test_acc = (test_pred == yh_test_t).float().mean().item()

                test_soft = torch.softmax(test_logits, dim=1)
                test_cos = nn.functional.cosine_similarity(test_soft, ys_test_t).mean().item()

                test_kl = nn.KLDivLoss(reduction='batchmean')(
                    torch.log_softmax(test_logits, dim=1), ys_test_t).item()

                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                history["test_kl"].append(test_kl)
                history["test_cos"].append(test_cos)

            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
                  f"test_cos={test_cos:.4f} test_kl={test_kl:.4f} "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  Training done ({elapsed:.0f}s)", flush=True)

    # ── 最终评估 ──
    print(f"\n[3/4] Final evaluation on test set...")
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_pred = test_logits.argmax(dim=1)
        final_acc = (test_pred == yh_test_t).float().mean().item()

        test_soft = torch.softmax(test_logits, dim=1)
        final_cos = nn.functional.cosine_similarity(test_soft, ys_test_t).mean().item()

        # Confusion matrix
        n_classes = 3
        conf_mat = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(len(yh_test_t)):
            conf_mat[yh_test_t[i].item(), test_pred[i].item()] += 1

    print(f"  Test Acc: {final_acc:.4f}")
    print(f"  Test Cosine Similarity: {final_cos:.4f}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>14} {'High-Dim':>10} {'Dense':>10} {'Uniform':>10}")
    conf_names = ["High-Dim", "Dense", "Uniform"]
    for i in range(n_classes):
        line = f"  {conf_names[i]:>12}"
        for j in range(n_classes):
            line += f" {conf_mat[i, j]:>10}"
        print(line)

    # Per-class accuracy
    print(f"\n  Per-class accuracy:")
    for i in range(n_classes):
        acc = conf_mat[i, i] / max(conf_mat[i].sum(), 1)
        print(f"    {conf_names[i]}: {acc:.4f} ({conf_mat[i, i]}/{conf_mat[i].sum()})")

    # ── 判定 ──
    print(f"\n[4/4] Decision:")
    print(f"  Acc_geo = {final_acc:.4f} (目标 > 0.75)")
    print(f"  Cos_sim = {final_cos:.4f} (目标 > 0.70)")

    if final_acc > 0.75:
        verdict = "✅ Acc_geo > 0.75 — 编码器能学会识别几何特性"
    elif final_acc > 0.55:
        verdict = "⚠️ 弱可预测 — 几何特性存在但难以精确分类"
    else:
        verdict = "❌ Acc_geo < 0.55 — 几何特性无法被学到"

    if final_cos > 0.70:
        verdict_soft = "✅ 余弦相似度高 — 连续几何得分可预测"
    elif final_cos > 0.50:
        verdict_soft = "⚠️ 连续得分部分可预测"
    else:
        verdict_soft = "❌ 连续得分无法预测"

    print(f"\n  Hard label: {verdict}")
    print(f"  Soft label: {verdict_soft}")

    # ── 保存结果 ──
    out_dir = "products/task16/from_task64"
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "config": {"epochs": args.epochs, "lr": args.lr, "hidden_dim": 128, "n_train": len(X_train), "n_test": len(X_test)},
        "final": {
            "test_accuracy": float(final_acc),
            "test_cosine_similarity": float(final_cos),
            "confusion_matrix": conf_mat.tolist(),
            "per_class_accuracy": {
                conf_names[i]: float(conf_mat[i, i] / max(conf_mat[i].sum(), 1))
                for i in range(n_classes)
            },
        },
        "history": history,
        "verdict": verdict,
        "verdict_soft": verdict_soft,
        "hard_label_feasible": final_acc > 0.75,
        "soft_label_feasible": final_cos > 0.70,
        "should_proceed": final_acc > 0.55 or final_cos > 0.50,
    }

    out_path = os.path.join(out_dir, "exp1_classifier_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Save model
    model_path = os.path.join(out_dir, "exp1_geo_predictor.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
