"""修复 Instruments train/valid/test jsonl — leave-one-out split bug.

Bug: 原生成脚本 train 和 valid 都用了相同的切片, 导致两者 inter_history 和 target_id 完全相同
     train.jsonl MD5 == valid.jsonl MD5 (f85ca5665d55b47de582a1d469f42168)

test.jsonl 格式 (参考):
  {"inter_history": ["1","2",...,"18"], "target_id": "19"}
  — inter_history 是 1-based item IDs (str), target_id 也是 1-based str

正确 leave-one-out (3 个独立文件):
  train: inter_history = 用户序列[:-2] (不含最后2个), target = 序列[-2]
  valid: inter_history = 用户序列[:-1] (不含最后1个), target = 序列[-1]
  test:  inter_history = 用户序列[:-1] (不含最后1个), target = 序列[-1]
  (3 个文件行数相同, 每行对应同一用户在不同 split 的不同 target)

inter.json: {user_id: [0-indexed int items]} (0-based embedding indices)
test.jsonl: inter_history 用 1-based str item IDs

转换: inter.json 的 0-based index → test.jsonl 的 1-based str
  emb_map: {str(idx): int(item_id_1based)}
  即 target_id_str = str(item_id_1based) = str(idx + 1)
"""
import json
import os
import hashlib

DATA_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Instruments"
inter_json = os.path.join(DATA_DIR, "Instruments.inter.json")

with open(inter_json) as f:
    inter = json.load(f)

# Sort users for deterministic order
user_ids = sorted(inter.keys(), key=lambda x: int(x))

train_data = []
valid_data = []

for user_id in user_ids:
    raw_seq = inter[user_id]  # list of 0-based int indices
    if len(raw_seq) < 3:
        continue

    # ETEGRec item2id uses 0-based embedding index as key
    # test.jsonl inter_history is 0-based (e.g. ['0','1','2']), target is int
    # Keep 0-based (matching test.jsonl format)
    def to_str(idx):
        return str(int(idx))

    # ETEGRec item2id uses 0-based embedding index as key.
    # HG-Rec 的原始 parquet 逻辑把每个用户的完整序列拆为：
    #   train parquet: seq[:-2]，随后 prepare_data 再取 history=[:-1], target=[-1]
    #                 => raw_seq[:-3] -> raw_seq[-2]
    #   valid parquet: seq[:-1]，随后 prepare_data => raw_seq[:-2] -> raw_seq[-1]
    #   test parquet:  seq[:]，随后 prepare_data => raw_seq[:-1] -> raw_seq[-1]
    # 三份 jsonl 必须与该语义一致；test.jsonl 已经是 raw_seq[:-1] -> raw_seq[-1]。

    # Train: raw_seq[:-3] as history, raw_seq[-3] as target.
    # This matches HG-Rec: train_data=seq[:-2], then prepare_data takes
    # history=[:-1] and target=[-1].
    train_hist = [to_str(x) for x in raw_seq[:-3]]
    train_target = to_str(raw_seq[-3])
    train_data.append({"inter_history": train_hist, "target_id": train_target})

    # Valid: raw_seq[:-2] as history, raw_seq[-2] as target.
    # This matches HG-Rec: val_data=seq[:-1], then prepare_data takes
    # history=[:-1] and target=[-1].
    valid_hist = [to_str(x) for x in raw_seq[:-2]]
    valid_target = to_str(raw_seq[-2])
    valid_data.append({"inter_history": valid_hist, "target_id": valid_target})

# Write
train_path = os.path.join(DATA_DIR, "Instruments.train.jsonl")
valid_path = os.path.join(DATA_DIR, "Instruments.valid.jsonl")

with open(train_path, "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(valid_path, "w") as f:
    for item in valid_data:
        f.write(json.dumps(item) + "\n")

print(f"Train: {len(train_data)} samples, {os.path.getsize(train_path):,} bytes")
print(f"Valid: {len(valid_data)} samples, {os.path.getsize(valid_path):,} bytes")

def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()

print(f"\nTrain MD5: {md5(train_path)}")
print(f"Valid MD5: {md5(valid_path)}")
print(f"Test MD5:  {md5(os.path.join(DATA_DIR, 'Instruments.test.jsonl'))}")

# Verify targets differ between train and valid at same row index
mismatch = sum(1 for i in range(min(len(train_data), len(valid_data)))
               if train_data[i]["target_id"] != valid_data[i]["target_id"])
print(f"\nTrain vs Valid target_id mismatch: {mismatch}/{len(train_data)} (should be {len(train_data)})")

# Verify format matches test.jsonl
with open(os.path.join(DATA_DIR, "Instruments.test.jsonl")) as f:
    test_sample = json.loads(f.readline())
print(f"\nTest format sample: {test_sample}")
print(f"Train format sample: {train_data[0]}")
