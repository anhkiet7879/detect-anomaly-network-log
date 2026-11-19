"""
log_template_extractor.py — [H3, H4] Bọc Drain3 thành bộ trích xuất template có Reset State.

Vì sao cần lớp bọc này?
  Drain3 "học" template dần dần: mỗi dòng mới có thể tạo template mới hoặc làm template cũ
  tổng quát hơn. Nếu để Drain3 tiếp tục học khi chạy trên tập Test, thông tin của tập Test sẽ
  "rò" vào bộ đặc trưng (data leakage). Lớp này tách rõ 2 pha:

    fit(train_messages)     -> reset() rồi HỌC template chỉ trên Train
    transform(messages)     -> CHỈ so khớp (match), KHÔNG sinh/sửa template;
                               dòng không khớp template nào -> template_id = -1, is_new_template = 1

  Sau fit(), cờ inference_only = True; gọi learn() lúc này sẽ báo lỗi.

Quy ước template_id (số nguyên):
   >= 1  : id cụm của Drain3 (chỉ có nghĩa TRONG một lần fit — fold khác có thể đánh số khác)
      0  : message rỗng (template "<EMPTY>"), không đưa vào Drain3
     -1  : không khớp template nào đã học (chỉ xuất hiện ở transform)

Cách dùng trong runner.py của Kiệt (mỗi fold):
    ext = LogTemplateExtractor("config/drain3.ini")
    train_df["template_id"] = ext.fit_transform(train_df["message"])
    test_ids, test_new = ext.transform(test_df["message"])
    test_df["template_id"], test_df["is_new_template"] = test_ids, test_new
"""
import time

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

EMPTY_ID = 0
NEW_ID = -1
EMPTY_TEMPLATE = "<EMPTY>"


class LogTemplateExtractor:
    def __init__(self, config_path="config/drain3.ini", full_search_strategy="fallback"):
        """
        full_search_strategy (dùng khi transform):
          "never"    chỉ tìm theo cây tiền tố (nhanh nhất, dễ báo "mới" nhầm hơn)
          "fallback" tìm theo cây trước, không thấy thì so với mọi template cùng độ dài (mặc định)
          "always"   luôn so với mọi template cùng độ dài (chậm nhất)
        """
        self.config_path = config_path
        self.full_search_strategy = full_search_strategy
        self.config = TemplateMinerConfig()
        self.config.load(config_path)
        self.config.profiling_enabled = False
        self.reset()

    # ------------------------------------------------------------ trạng thái
    def reset(self):
        """Khởi tạo lại cây tiền tố từ đầu (gọi ở đầu mỗi fold)."""
        self.miner = TemplateMiner(persistence_handler=None, config=self.config)
        self.inference_only = False
        self.n_empty_train = 0
        self._match_cache = {}          # message -> template_id, chỉ dùng khi đã khóa
        self.fit_seconds = None

    @property
    def n_templates(self):
        """Số template đã học (không tính <EMPTY>)."""
        return len(self.miner.drain.clusters)

    def fingerprint(self):
        """Dấu vân tay của từ điển: (số template, {id: template}).
        Dùng cho test T4: phải KHÔNG đổi sau transform()."""
        return self.n_templates, {c.cluster_id: c.get_template() for c in self.miner.drain.clusters}

    # ------------------------------------------------------------ học (chỉ Train)
    def learn(self, message):
        """Học một dòng. Chỉ được gọi khi chưa khóa."""
        if self.inference_only:
            raise RuntimeError("Extractor đang ở chế độ inference_only — không được học thêm template "
                               "(gọi reset() hoặc fit() để học lại từ đầu).")
        if not message:
            self.n_empty_train += 1
            return EMPTY_ID
        return self.miner.add_log_message(message)["cluster_id"]

    def fit_transform(self, messages):
        """reset() -> học trên toàn bộ messages -> khóa. Trả về mảng template_id của chính các dòng đó."""
        self.reset()
        t0 = time.perf_counter()
        ids = np.fromiter((self.learn(m) for m in messages), dtype=np.int32)
        self.fit_seconds = time.perf_counter() - t0
        self.inference_only = True
        self._train_counts = pd.Series(ids).value_counts()
        return ids

    def fit(self, messages):
        self.fit_transform(messages)
        return self

    # ------------------------------------------------------------ suy luận (Test / thực tế)
    def match_one(self, message):
        if not message:
            return EMPTY_ID
        tid = self._match_cache.get(message)
        if tid is None:
            c = self.miner.match(message, full_search_strategy=self.full_search_strategy)
            tid = c.cluster_id if c is not None else NEW_ID
            self._match_cache[message] = tid     # an toàn vì từ điển đã khóa
        return tid

    def transform(self, messages):
        """Gán template cho dòng mới mà KHÔNG sửa từ điển.
        Trả về (template_id, is_new_template) — 2 mảng numpy cùng độ dài với messages."""
        if not self.inference_only:
            raise RuntimeError("Chưa fit() — hãy học trên tập Train trước.")
        ids = np.fromiter((self.match_one(m) for m in messages), dtype=np.int32)
        return ids, (ids == NEW_ID).astype(np.int8)

    # ------------------------------------------------------------ xuất từ điển
    def templates(self):
        """Bảng template đã học: template_id, train_count, template (sắp theo train_count giảm dần).
        train_count = số dòng Train thuộc template -> Phước dùng để xét template hiếm (P2)."""
        rows = [(c.cluster_id, c.get_template()) for c in self.miner.drain.clusters]
        if self.n_empty_train:
            rows.append((EMPTY_ID, EMPTY_TEMPLATE))
        df = pd.DataFrame(rows, columns=["template_id", "template"])
        counts = getattr(self, "_train_counts", pd.Series(dtype="int64"))
        df["train_count"] = df["template_id"].map(counts).fillna(0).astype("int64")
        return (df[["template_id", "train_count", "template"]]
                .sort_values("train_count", ascending=False).reset_index(drop=True))
