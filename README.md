# HỌC KỲ DOANH NGHIỆP UTH – VNPT NET
## ĐỀ TÀI N3.10: PHÁT HIỆN BẤT THƯỜNG TRONG LOG VÀ DỮ LIỆU VẬN HÀNH MẠNG
**Định hướng kỹ thuật:** AIOps, Observability, Kiến trúc Microservices (Grafana + Loki + PostgreSQL + FastAPI trên Local Docker)

---

### Thành viên nhóm
| STT | Họ và tên | Vai trò | Mã Task |
|---|---|---|---|
| 1 | **Anh Kiệt** | Leader & System Architect | K1 – K21 |
| 2 | **Huy** | Code chính (Parser, Drain3, ML, Loki, API) | H1 – H12 |
| 3 | **Phước** | Code phụ (Rule, Z-score, DB, Grafana, Docker) | P1 – P12 |
| 4 | **Duy Khang** | Testing độc lập, Dataset giả lập & Báo cáo Word | T1 – T12 |

---

### Cấu trúc thư mục dự án
```text
N3.10/
├── config/                  # Cấu hình tham số hóa YAML (experiment.yaml, drain3.ini, ...)
├── data/                    # Dữ liệu (chỉ commit file mẫu/quarantine, không commit file raw/parquet lớn)
│   ├── raw/                 # [Git ignored] Dữ liệu thô BGL.log, OpenSSH, TelecomTS
│   └── processed/           # Dữ liệu sau chuẩn hóa
├── docs/                    # Tài liệu đặc tả kỹ thuật, schema, ADR kiến trúc
├── reports/                 # Báo cáo đánh giá chất lượng dữ liệu và kết quả thực nghiệm
├── scripts/                 # Các script tiện ích kiểm tra dữ liệu
├── src/                     # Toàn bộ mã nguồn chính của hệ thống
│   ├── alert/               # Alert Factory, Deduplication, Severity mapping (Kiệt)
│   ├── evaluation/          # Khung chia tập dữ liệu Zero-Leakage & Runner (Kiệt)
│   ├── models/              # Isolation Forest, Z-score, Rule detectors (Huy, Phước)
│   ├── parser/              # Parser chuẩn hóa log sang Common Log Schema (Huy)
│   └── service/             # FastAPI backend (Huy)
├── tests/                   # Bộ kiểm thử tự động pytest (Khang)
└── README.md
```

---

### Nguyên tắc Git & Phối hợp
1. Nhánh `main` là nhánh bảo vệ, chỉ merge qua **Pull Request (PR)** có ít nhất 1 thành viên review.
2. Không commit file dữ liệu nặng (> 50MB) lên repo. Dữ liệu thô tải cục bộ vào thư mục `data/raw/`.
3. Tuân thủ quy ước đặt tên nhánh:
   - `feat/<task-id>-<tên-ngắn-gọn>` (ví dụ: `feat/k1-data-splitter`)
   - `test/<task-id>-<tên-ngắn-gọn>` (ví dụ: `test/t4-leakage-pytest`)
