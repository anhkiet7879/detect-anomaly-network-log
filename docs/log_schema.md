# COMMON LOG SCHEMA — CHUẨN 12 TRƯỜNG LOG THỐNG NHẤT
**Mã Task:** K2 (P0) — Leader & Architect  
**Đề tài:** N3.10 – Phát hiện bất thường trong log và dữ liệu vận hành mạng  
**Ngày chốt:** 20/11/2026  
**Trạng thái:** ✅ CHỐT — Mọi parser mới phải tuân thủ schema này

---

## 1. Mục tiêu

Định nghĩa một **bộ trường dữ liệu chung (Common Log Schema)** mà mọi parser (BGL, OpenSSH, TelecomTS, và dữ liệu VNPT tương lai) phải xuất ra — bất kể định dạng log gốc khác nhau như thế nào. Nhờ vậy:
- Toàn bộ pipeline phía sau (cửa sổ trượt, Drain3, 3 detector, Alert Engine, Grafana) **chỉ cần đọc một cấu trúc duy nhất**.
- Thêm nguồn log mới chỉ cần viết thêm 1 parser, không cần sửa pipeline.

---

## 2. Schema 12 cột chính thức

> **Quy ước:** Tên cột bằng **snake_case**, kiểu dữ liệu tương thích CSV / Parquet / PostgreSQL.

| # | Tên cột | Kiểu | Bắt buộc | Mô tả | Ví dụ (BGL) | Ví dụ (OpenSSH) |
|---|---------|------|----------|-------|-------------|-----------------|
| 1 | `line_no` | int64 | ✅ | Số thứ tự dòng trong file gốc (đếm từ 0) | `0` | `0` |
| 2 | `source` | string | ✅ | Mã định danh nguồn dữ liệu (viết thường, không dấu) | `"bgl"` | `"openssh"` |
| 3 | `ts_epoch` | int64 | ✅ | Mốc thời gian UTC tính bằng **giây** (Unix epoch). Đây là trục thời gian chính dùng để sắp xếp, cắt cửa sổ, và chia train/test. | `1117869872` | `1134190546` |
| 4 | `ts_utc` | string | ✅ | Mốc thời gian UTC dạng ISO 8601 có hậu tố `Z` — chỉ dùng cho con người đọc / Grafana hiển thị. | `"2005-06-04T07:24:32Z"` | `"2005-12-10T06:55:46Z"` |
| 5 | `node` | string | ✅ | Vị trí nguồn phát (hostname, địa chỉ phần cứng, IP, ...). Nếu log không có thông tin node: điền tên host hoặc `"unknown"`. | `"R02-M1-N0-C:J12-U11"` | `"LabSZ"` |
| 6 | `component` | string | ✅ | Bộ phận / dịch vụ / tiến trình phát sinh log. | `"KERNEL"` | `"sshd"` |
| 7 | `log_type` | string | ⬚ | Phân loại phụ theo chuẩn riêng của nguồn (VD: `RAS` trong BGL). Để trống `""` nếu nguồn không có. | `"RAS"` | `""` |
| 8 | `severity_raw` | string | ✅ | Mức nghiêm trọng nguyên văn từ log gốc, không chuẩn hóa. | `"FATAL"` | `"error"` |
| 9 | `severity_norm` | int (0–7) | ⬚ | Mức nghiêm trọng đã ánh xạ sang thang syslog RFC 5424 (0=Emergency … 7=Debug). Để trống nếu không ánh xạ được. | `2` (Critical) | `3` (Error) |
| 10 | `message` | string | ✅ | Nội dung chính của dòng log, nguyên văn. Có thể rỗng. Đây là đầu vào cho Drain3 (`LogTemplateExtractor`). | `"data TLB error interrupt"` | `"Failed password for invalid user webmaster from 173.234.31.186 port 38926 ssh2"` |
| 11 | `gt_alert_type` | string | ⬚ | **Nhãn chân lý** — Tên loại bất thường. Điền `"NORMAL"` nếu dòng bình thường (không để ô rỗng để tránh pandas parse thành NaN). **KHÔNG BAO GIỜ** dùng làm đặc trưng mô hình. | `"KERNDTLB"` | `"BRUTEFORCE"` |
| 12 | `gt_is_alert` | int8 (0/1) | ⬚ | **Nhãn chân lý** — 1 = bất thường, 0 = bình thường. **KHÔNG BAO GIỜ** dùng làm đặc trưng mô hình. | `1` | `0` |

---

## 3. Quy tắc bắt buộc cho mọi Parser

### 3.1. Chuẩn hóa thời gian
- **`ts_epoch` PHẢI là giây UTC** (int64, không phải mili giây).
- **`ts_utc` PHẢI có hậu tố `Z`** (không dùng `+00:00` hay offset múi giờ địa phương).
- Nếu log gốc không ghi năm (ví dụ OpenSSH: `Dec 10 06:55:46`), parser phải suy luận năm hợp lý hoặc yêu cầu tham số `--year`.

### 3.2. Bảng ánh xạ Severity (severity_raw → severity_norm)

| severity_norm | Tên syslog | BGL | OpenSSH | TelecomTS |
|:---:|---|---|---|---|
| 0 | Emergency | — | — | — |
| 1 | Alert | — | — | — |
| 2 | Critical | `FATAL`, `FAILURE` | — | — |
| 3 | Error | `ERROR`, `SEVERE` | `error` | — |
| 4 | Warning | `WARNING` | `warning` | — |
| 5 | Notice | — | `notice`, `authpriv.info` | — |
| 6 | Info | `INFO` | `info` | — |
| 7 | Debug | — | `debug` | — |

> Parser cho TelecomTS (dữ liệu metric dạng time series, không phải log text) sẽ được xử lý riêng bằng adapter chuyển đổi metric → schema (xem mục 5).

### 3.3. Phân cách trường chân lý (Ground Truth)
- Cột 11 (`gt_alert_type`) và cột 12 (`gt_is_alert`) là **nhãn đáp án** dùng cho đánh giá.
- **Quy tắc bất biến:** Không pipeline nào, không detector nào được phép đọc 2 cột này làm đầu vào đặc trưng. Chỉ sử dụng trong `runner.py` (K1) và `evaluate.py` (Khang – T4) để tính điểm.

### 3.4. Quarantine (Cách ly dòng lỗi)
- Dòng log không parse được (sai định dạng, thiếu trường, encoding lỗi) phải được ghi vào file cách ly `*_quarantine.txt` với định dạng: `<line_no>\t<nội_dung_nguyên_văn>`.
- **Tuyệt đối không loại bỏ âm thầm** — cần biết bao nhiêu % dữ liệu bị mất.

---

## 4. Các Parser hiện tại và trạng thái tuân thủ

| Nguồn | Parser | Trạng thái | Ghi chú |
|---|---|---|---|
| **BGL** | `bgl_parse_v2.py` (Huy) | ✅ Tuân thủ 12 cột | Đã chạy trên 949k dòng. Xuất CSV + quarantine. |
| **OpenSSH** | *Chưa có* | 🔲 Cần viết (Task K19/H9) | Log dạng syslog, cần xử lý PID (`sshd[24200]`), thiếu năm. |
| **TelecomTS** | *Adapter riêng* | 🔲 Cần viết (Task K20/H10) | Dữ liệu metric (CSV 19 cột số), không phải log text. Xem mục 5. |

---

## 5. Adapter cho TelecomTS (Dữ liệu metric / time series)

TelecomTS không phải log text mà là **chuỗi metric mạng** (RSRP, BLER, MCS, PRB, throughput, ...) đo mỗi 100ms. Để tương thích với pipeline chung, cần viết adapter chuyển đổi:

| Trường schema | Giá trị cho TelecomTS |
|---|---|
| `source` | `"telecomts"` |
| `ts_epoch` | Chuyển đổi từ cột `timestamp` (ví dụ: `2036-12-10 00:00:00.100`) |
| `node` | Zone + loại thiết bị (VD: `"Zone_A/stationary"`) |
| `component` | Loại traffic (VD: `"File"`, `"Twitch"`, `"YouTube"`) |
| `severity_raw` | `"metric"` (không có severity truyền thống) |
| `message` | Chuỗi JSON hoặc CSV inline chứa toàn bộ metric (RSRP, BLER, ...) |
| `gt_alert_type` | `"jammer"`, `"synthetic"`, hoặc `"NORMAL"` (theo label thư mục) |
| `gt_is_alert` | 1 nếu thuộc thư mục `anomalous/`, 0 nếu thuộc `normal/` |

---

## 6. Hợp đồng giao tiếp với các module phía sau

| Module phía sau | Cột schema cần dùng | Người phụ trách |
|---|---|---|
| **Drain3** (`LogTemplateExtractor`) | `message` | Huy (H3) |
| **Window Features** (`window_features.py`) | `ts_epoch`, `node`, `component`, `severity_raw`, `message`, `gt_is_alert`, `gt_alert_type` | Huy (H5) |
| **Splitter** (`splitter.py`) | `ts_epoch`, nhãn `y` (tính từ `gt_is_alert` ở tầng window) | Kiệt (K1) |
| **Alert Engine** (`alert_factory.py`) | `ts_epoch`, `ts_utc`, `node`, `component`, `severity_raw`, `source` | Kiệt (K5) |
| **PostgreSQL / Grafana** | Toàn bộ 12 cột (đẩy vào bảng `logs`) | Phước (P5) |
