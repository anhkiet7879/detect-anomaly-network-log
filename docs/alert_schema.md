# ALERT SCHEMA — CẤU TRÚC CẢNH BÁO THỐNG NHẤT 3 TẦNG DETECTOR
**Mã Task:** K3 (P0) — Leader & Architect  
**Đề tài:** N3.10 – Phát hiện bất thường trong log và dữ liệu vận hành mạng  
**Ngày chốt:** 20/11/2026  
**Trạng thái:** ✅ CHỐT — Mọi detector phải xuất Alert theo cấu trúc này

---

## 1. Mục tiêu

Mỗi tầng detector (Luật nghiệp vụ, Baseline Z-score, Isolation Forest) có logic rất khác nhau, nhưng đầu ra cần đồng nhất để:
- **Alert Engine** (`src/alert/alert_factory.py` — K5) gộp, xếp hạng và chống bão cảnh báo.
- **Grafana Dashboard** hiển thị cùng bảng annotation/panel.
- **PostgreSQL** lưu trữ trong 1 bảng `alerts` duy nhất.
- **Playbook Engine** (K8–K10) tra cứu cách xử lý tự động theo `alert_type`.

---

## 2. Alert Schema — 16 trường

Mỗi detector khi phát hiện bất thường trên 1 cửa sổ 5 phút sẽ tạo 1 bản ghi Alert theo cấu trúc sau:

| # | Tên trường | Kiểu | Bắt buộc | Mô tả |
|---|-----------|------|----------|-------|
| 1 | `alert_id` | string (UUID4) | ✅ | Định danh duy nhất của cảnh báo. Sinh tự động bởi Alert Factory. |
| 2 | `ts_epoch` | int64 | ✅ | Mốc **đầu** cửa sổ 5 phút gây cảnh báo (epoch giây UTC). |
| 3 | `ts_utc` | string | ✅ | ISO 8601 UTC, VD: `"2005-06-14T16:05:00Z"`. |
| 4 | `source` | string | ✅ | Nguồn dữ liệu: `"bgl"`, `"openssh"`, `"telecomts"`. |
| 5 | `detector` | string | ✅ | Tên bộ phát hiện đã tạo cảnh báo này. |
| 6 | `alert_type` | string | ✅ | Mã loại cảnh báo (tra Playbook). Xem bảng mã ở mục 3. |
| 7 | `severity` | string | ✅ | Mức nghiêm trọng: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`. Xem bảng ánh xạ ở mục 4. |
| 8 | `severity_score` | int (1–4) | ✅ | Giá trị số tương ứng severity (1=LOW … 4=CRITICAL), dùng để sắp xếp. |
| 9 | `score` | float | ✅ | Điểm bất thường do detector tính (ý nghĩa tùy loại detector). |
| 10 | `threshold` | float | ✅ | Ngưỡng mà detector đã dùng để so sánh với `score`. |
| 11 | `title` | string | ✅ | Tiêu đề ngắn gọn cho con người đọc nhanh (~80 ký tự). |
| 12 | `description` | string | ✅ | Mô tả chi tiết bao gồm bối cảnh cửa sổ. |
| 13 | `evidence` | object (JSON) | ✅ | Bằng chứng kỹ thuật. Cấu trúc chi tiết ở mục 5. |
| 14 | `playbook_id` | string | ⬚ | Mã Playbook áp dụng (VD: `"PB-001"`). Để trống nếu chưa gán. |
| 15 | `dedup_key` | string | ✅ | Khóa chống trùng lặp: `"{source}:{detector}:{alert_type}:{ts_epoch}"`. Alert Engine dùng khóa này để gộp bão cảnh báo. |
| 16 | `created_at` | string | ✅ | Mốc thời gian thực tế hệ thống tạo alert (ISO 8601 UTC). |

---

## 3. Bảng mã loại cảnh báo (`alert_type`)

| `alert_type` | Detector | Ý nghĩa | Playbook |
|---|---|---|---|
| `RULE_FATAL_LOG` | Rule (Phước – P1) | Cửa sổ có ≥1 dòng log FATAL | PB-001 |
| `RULE_ERROR_SPIKE` | Rule (Phước – P1) | Số dòng ERROR vượt ngưỡng tĩnh | PB-001 |
| `ZSCORE_LOG_VOLUME` | Z-score (Phước – P3) | Tổng số log/5 phút có z-score > ngưỡng | PB-002 |
| `ZSCORE_SEVERITY_SPIKE` | Z-score (Phước – P3) | Tổng log FATAL+ERROR có z-score > ngưỡng | PB-002 |
| `IF_MULTIVARIATE` | Isolation Forest (Huy – H7) | Điểm bất thường đa chiều vượt ngưỡng phân vị | PB-004 |
| `IF_TEMPLATE_DRIFT` | Isolation Forest (Huy – H7) | Phát hiện khi kết hợp đặc trưng template mới/hiếm | PB-004 |
| `NEW_TEMPLATE` | Rule + Drain3 | Cửa sổ xuất hiện template log chưa từng thấy khi học | PB-003 |
| `RARE_TEMPLATE_BURST` | Rule + Drain3 | Template hiếm (< 10 dòng trên Train) tăng đột biến | PB-003 |

---

## 4. Bảng ánh xạ Severity (K6 — Tiền định nghĩa)

| Mức | Giá trị | Tiêu chí |
|---|:---:|---|
| `CRITICAL` | 4 | Có log FATAL + Isolation Forest phát hiện đồng thời, HOẶC cửa sổ nằm trong đợt sự cố kéo dài > 10 phút |
| `HIGH` | 3 | Isolation Forest vượt ngưỡng (phân vị 99%) HOẶC z-score > 5 HOẶC template mới + hơn 50 dòng |
| `MEDIUM` | 2 | Rule FATAL đơn lẻ HOẶC z-score > 3 HOẶC template hiếm bất ngờ |
| `LOW` | 1 | Z-score 2–3 HOẶC template mới nhưng ít dòng (< 5) |

> **Lưu ý:** Khi nhiều detector cùng cảnh báo trên 1 cửa sổ, Alert Engine chọn severity **cao nhất** làm severity cuối cùng của cửa sổ đó.

---

## 5. Cấu trúc Evidence (Bằng chứng kỹ thuật — K8)

Trường `evidence` chứa bối cảnh giúp kỹ sư vận hành hiểu **tại sao** hệ thống cảnh báo, không cần mở log thô:

```json
{
  "window": {
    "ts_epoch": 1118765100,
    "ts_utc": "2005-06-14T16:05:00Z",
    "duration_sec": 300,
    "total_logs": 16422,
    "total_alerts_gt": 16384
  },
  "features": {
    "f_n_logs": 16422,
    "f_n_nodes": 512,
    "f_sev_FATAL": 16384,
    "f_n_new_tpl_lines": 0,
    "f_n_rare_tpl_lines": 38
  },
  "detector_detail": {
    "method": "isolation_forest",
    "score": 0.892,
    "threshold": 0.634,
    "contamination": 0.01,
    "top_features": [
      {"name": "f_sev_FATAL", "value": 16384, "contribution": 0.72},
      {"name": "f_n_logs", "value": 16422, "contribution": 0.15},
      {"name": "f_n_nodes", "value": 512, "contribution": 0.08}
    ]
  },
  "context": {
    "top_templates": [
      {"template_id": 122, "template": "data storage interrupt", "count": 16384}
    ],
    "top_nodes": ["R02-M1-N0-C:J12-U01", "R02-M1-N0-C:J12-U11"],
    "top_components": ["KERNEL"],
    "prev_window_alert": false,
    "next_window_alert": true
  }
}
```

### Mô tả từng khối:

| Khối | Mô tả | Bắt buộc |
|---|---|---|
| `window` | Thông tin tổng quan cửa sổ: mốc thời gian, số dòng log, số dòng bất thường (ground-truth). | ✅ |
| `features` | Giá trị các đặc trưng chính tại cửa sổ này (snapshot). | ✅ |
| `detector_detail` | Chi tiết của detector: điểm, ngưỡng, phương pháp, và top 3 đặc trưng đóng góp lớn nhất (nếu tính được). | ✅ |
| `context` | Bối cảnh xung quanh: template phổ biến nhất, node bị ảnh hưởng, cửa sổ kế cận có cảnh báo không. | ⬚ |

---

## 6. Hợp đồng giao tiếp — Ai tạo Alert, Ai tiêu thụ Alert

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Rule Engine │    │  Z-score     │    │  Isolation   │
│  (Phước P1)  │    │  (Phước P3)  │    │  Forest (H7) │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │ Alert             │ Alert             │ Alert
       │ (schema 16 cột)   │ (schema 16 cột)   │ (schema 16 cột)
       └─────────┬─────────┴─────────┬─────────┘
                 ▼                   ▼
         ┌───────────────────────────────┐
         │     ALERT ENGINE (Kiệt K5)   │
         │  - Gộp trùng (dedup_key)     │
         │  - Chọn severity cao nhất    │
         │  - Gán playbook_id           │
         ├───────────────────────────────┤
         │          Đầu ra              │
         ├──────┬──────────┬────────────┤
         │      ▼          ▼            ▼
         │  PostgreSQL   Grafana     Playbook
         │  (Phước P5)   Dashboard   (Kiệt K10)
         └──────────────────────────────┘
```

---

## 7. Ví dụ Alert đầy đủ (JSON)

```json
{
  "alert_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "ts_epoch": 1118765100,
  "ts_utc": "2005-06-14T16:05:00Z",
  "source": "bgl",
  "detector": "Isolation_Forest",
  "alert_type": "IF_MULTIVARIATE",
  "severity": "CRITICAL",
  "severity_score": 4,
  "score": 0.892,
  "threshold": 0.634,
  "title": "[CRITICAL] BGL cửa sổ 16:05 — Bất thường đa chiều (IF score 0.892 > 0.634)",
  "description": "Isolation Forest phát hiện 16,422 dòng log trong cửa sổ 5 phút 16:05–16:10 UTC ngày 2005-06-14, gồm 16,384 dòng FATAL severity từ KERNEL/data storage interrupt trên 512 node. Điểm bất thường 0.892 vượt ngưỡng 0.634.",
  "evidence": { "...xem mục 5..." },
  "playbook_id": "PB-004",
  "dedup_key": "bgl:Isolation_Forest:IF_MULTIVARIATE:1118765100",
  "created_at": "2026-11-20T02:00:00Z"
}
```
