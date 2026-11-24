CHECK_1已重新打开issue正文，SHA不变。17项：2PASS、2FAIL、13NOT_RUN，整体FAIL，不派发正式训练。C10为真实CUDA数值失败，C13为真实60epoch inclusive0.91185s>0.8；新kernel/同一步复用已实现且CPU10tests通过，CUDA GREEN及真实loss/四rank完整准入尚待执行。

证据 `results/research/issue253/CHECK_1.md`；下一轮修GPU数值与完整性能，不删2-hop/采样，不放宽阈值。
