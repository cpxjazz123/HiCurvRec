新增R4 FAIL：`python3.9 -m tools.issue256.admission` exit1，behavior_graph=PASS，但真实entry失败，禁止派发。错误原文：`train_rqvae.py:383 calibration_ids = _calibration_ids(train_ids, device, train_sampler)` → `_calibration_ids:255 ValueError: calibration requires the actual distributed training sampler`。完整证据`rounds/r01/ADMISSION.json`；框架entry是合法单进程smoke，train_sampler=None，之前四卡诊断没有覆盖它。

这是可修复的入口兼容故障，不是科学负结果/外部阻塞。不会绕过entry或改闸门。按已冻结“实际首batch、不得推进随机状态”契约，改为统一从真实DataLoader窥视首batch并恢复RNG/显式generator状态；分布式和非分布式都用同一接口。先补None/RandomSampler复现与顺序/RNG回归，再修实现；旧121/64测试、DDP/性能凭据与失败记录按源代保存，改源码后重做当前凭据。
