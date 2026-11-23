# QUY TẮC ÁNH XẠ MỨC ĐỘ NGHIÊM TRỌNG (SEVERITY MAPPING RULES)
**Mã Task:** K6 (P0) — Leader & Architect  
**Đề tài:** N3.10 – Phát hiện bất thường trong log và dữ liệu vận hành mạng  
**Cấu hình thực thi:** `config/severity_rules.yaml` | **Mã nguồn:** `src/alert/severity.py`  
**Ngày chốt:** 20/11/2026  

---

## 1. Mục tiêu

Chuẩn hóa cách phân loại mức độ nghiêm trọng của mọi cảnh báo từ 3 tầng detector (**Luật nghiệp vụ, Baseline Z-score, Isolation Forest**) về thang đo thống nhất 4 cấp từ `LOW` đến `CRITICAL`, giúp kỹ sư NOC/SOC ưu tiên xử lý sự cố đúng trọng tâm và hỗ trợ lọc/sắp xếp trên Grafana.

---

## 2. Thang đo 4 cấp độ Severity

| Mức độ (`severity`) | Điểm số (`severity_score`) | Màu đề xuất (Grafana) | SLA phản ứng khuyến nghị |
|---|:---:|---|---|
| **CRITICAL** | `4` | Đỏ (`#E02F44`) | Xử lý ngay lập tức (< 5 phút) — sự cố nghiêm trọng / lan rộng |
| **HIGH** | `3` | Cam (`#FF780A`) | Điều tra gấp (< 15 phút) — bất thường đa chiều hoặc tăng đột biến mạnh |
| **MEDIUM** | `2` | Vàng (`#FADE2A`) | Theo dõi & kiểm tra (< 60 phút) — lỗi đơn lẻ hoặc tăng lưu lượng đáng chú ý |
| **LOW** | `1` | Xanh dương (`#5794F2`) | Ghi nhận bối cảnh / cảnh báo sớm |

---

## 3. Bảng quy tắc ánh xạ cơ sở theo loại cảnh báo (`alert_type`)

| `alert_type` | `LOW (1)` | `MEDIUM (2)` | `HIGH (3)` | `CRITICAL (4)` |
|---|---|---|---|---|
| `RULE_FATAL_LOG` | — | `1 <= f_sev_FATAL < 100` | `100 <= f_sev_FATAL < 1000` | `f_sev_FATAL >= 1000` (bão FATAL) |
| `RULE_ERROR_SPIKE` | — | `threshold < score < 2*threshold` | `2*threshold <= score < 5*threshold` | `score >= 5*threshold` |
| `ZSCORE_LOG_VOLUME` | `2.0 <= z <= 3.0` | `3.0 < z <= 5.0` | `5.0 < z < 10.0` | `z >= 10.0` |
| `ZSCORE_SEVERITY_SPIKE` | `2.0 <= z <= 3.0` | `3.0 < z <= 5.0` | `5.0 < z < 8.0` | `z >= 8.0` |
| `IF_MULTIVARIATE` | — | `score < threshold` (fallback) | `score >= threshold` (phân vị 99%) | Khi kết hợp luật leo thang (Mục 4) |
| `IF_TEMPLATE_DRIFT` | — | `score < threshold` (fallback) | `score >= threshold` | `f_n_new_tpl_lines >= 200` hoặc leo thang |
| `NEW_TEMPLATE` | `f_n_new_tpl_lines < 5` | `5 <= f_n_new_tpl_lines <= 50` | `f_n_new_tpl_lines > 50` | Khi kết hợp luật leo thang |
| `RARE_TEMPLATE_BURST` | — | `f_n_rare_tpl_lines < 50` | `50 <= f_n_rare_tpl_lines < 500` | `f_n_rare_tpl_lines >= 500` |

---

## 4. Quy tắc Leo thang (Escalation Rules) lên `CRITICAL (4)`

Sau khi tính mức cơ sở, hệ thống luôn áp dụng 3 quy tắc leo thang (thực thi ở cả `AlertFactory` K5 và `AlertDeduplicator` K7):

1. **Xác nhận chéo FATAL + Isolation Forest (`ESCALATION_FATAL_PLUS_IF`):**
   - Nếu trên cùng cửa sổ 5 phút vừa có log mức `FATAL` (`f_sev_FATAL > 0` hoặc `RULE_FATAL_LOG`) **vừa** bị `Isolation Forest` (`IF_MULTIVARIATE` / `IF_TEMPLATE_DRIFT`) gắn cờ $\rightarrow$ tự động nâng lên `CRITICAL (4)`.
2. **Đợt sự cố kéo dài (`ESCALATION_PROLONGED_INCIDENT`):**
   - Khi module gộp cảnh báo lặp (`dedup.py` – K7) phát hiện detector cảnh báo liên tiếp $\ge 3$ cửa sổ (15 phút) HOẶC tổng thời lượng vượt quá **10 phút** (`duration_sec > 600`) $\rightarrow$ tự động nâng lên `CRITICAL (4)` (tuân thủ nguyên tắc Zero Data Leakage).
3. **Đồng thuận đa tầng (`ESCALATION_MULTI_DETECTOR`):**
   - Nếu có $\ge 3$ detector độc lập cùng kích hoạt cảnh báo trên cùng cửa sổ 5 phút $\rightarrow$ tự động nâng lên `CRITICAL (4)`.

> **Nguyên tắc bất biến khi hợp nhất:** Khi nhiều cảnh báo được gộp lại, mức `severity` cuối cùng luôn bằng $\max(\text{severity}_{\text{thành phần}}, \text{severity}_{\text{leo thang}})$ — tuyệt đối không bao giờ hạ cấp cảnh báo nghiêm trọng.

---

## 5. Ví dụ áp dụng thực tế trên dữ liệu BGL

| Tình huống trên BGL | Đầu vào | Kết quả Severity | Luật khớp (`matched_rule`) |
|---|---|---|---|
| Xuất hiện 1 dòng `data TLB error interrupt` đơn lẻ | `RULE_FATAL_LOG`, `f_sev_FATAL=1` | `MEDIUM (2)` | `RULE_FATAL_SINGLE` |
| Cửa sổ có 3 dòng template mới chưa từng gặp | `NEW_TEMPLATE`, `f_n_new_tpl_lines=3` | `LOW (1)` | `NEW_TEMPLATE_MINOR` |
| Z-score tổng lượng log đạt `6.2` | `ZSCORE_LOG_VOLUME`, `score=6.2` | `HIGH (3)` | `ZSCORE_LOG_VOLUME_HIGH` |
| Isolation Forest chấm `0.892 > 0.634` trên cửa sổ có `16,384` dòng `FATAL` | `IF_MULTIVARIATE`, `f_sev_FATAL=16384` | `CRITICAL (4)` | `ESCALATION_FATAL_PLUS_IF` |
| Đợt lỗi `KERNDTLB` lặp liên tục qua 4 cửa sổ 5 phút (15 phút) | `RULE_FATAL_LOG` gộp bởi `dedup.py`, `duration_sec=900` | `CRITICAL (4)` | `ESCALATION_PROLONGED_INCIDENT` |
