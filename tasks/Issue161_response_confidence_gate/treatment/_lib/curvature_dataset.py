"""Issue #159: 曲率状态 dataset — 为每个 history/target item 附加其 Stage2 有效曲率.

每个 item 的 4-digit SID 唯一确定其 item id (code_to_item), 从而查 curvature_state
(9922×3, 每 item 每层有效 c_l,i). 序列保持 item 边界, 每个 code token 对应该层曲率.
"""

import numpy as np
import torch
from dataset import GenRecDataset, process_data


class CurvatureGenRecDataset(GenRecDataset):
    def __init__(self, dataset_path, code_path, curvature_path, mode, codebook_size,
                 max_len, PAD_TOKEN=0):
        self.curvature_state = np.load(curvature_path)  # (n_items, 3) 每 item 每层有效 c
        super().__init__(dataset_path, code_path, mode, codebook_size, max_len, PAD_TOKEN)

    def _prepare_data(self):
        """与 GenRecDataset._prepare_data 相同 (history→code), 仅附加曲率状态."""
        processed = process_data(self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN)
        n_codes = self.curvature_state.shape[0]
        for item in processed:
            # 与 base 相同: history item ids → 4-digit code; 同时附加曲率
            hist_items = list(item['history'])
            hist_codes = []
            hist_c = []
            hist_s = []
            for x in hist_items:
                code = self.item_to_code.get(x, np.array([self.PAD_TOKEN] * 4))
                hist_codes.append(code)
                c = np.ones(4, dtype=np.float32)  # 默认 1.0 (无效 item)
                s_vec = np.ones(4, dtype=np.float32)
                if 1 <= x <= n_codes:
                    cs = self.curvature_state[x - 1]
                    c = np.array([cs[0], cs[1], cs[2], cs[2]], dtype=np.float32)
                    # Issue #161: 响应特征 (只读曲率) — [c, boundary, |dc_L0L1|, |dc_L1L2|]
                    s_vec = np.array([
                        cs[0],
                        1.0 - cs[1] / 2.0,
                        abs(cs[0] - cs[1]),
                        abs(cs[1] - cs[2]),
                    ], dtype=np.float32)
                hist_c.append(c)
                hist_s.append(np.tile(s_vec, 4))  # 每 item 4 token, 每 token 4 维
            item['history'] = hist_codes
            t = item['target']
            target_code = self.item_to_code.get(t, np.array([self.PAD_TOKEN] * 4))
            item['target'] = target_code
            c = np.ones(4, dtype=np.float32)
            if 1 <= t <= n_codes:
                cs = self.curvature_state[t - 1]
                c = np.array([cs[0], cs[1], cs[2], cs[2]], dtype=np.float32)
            item['target_c'] = c
            item['history_c'] = hist_c
            item['history_s'] = hist_s
        return processed
