# TÀI LIỆU KHUNG THỰC NGHIỆM VÀ ĐÁNH GIÁ (EVALUATION FRAMEWORK)
**Đề tài:** N3.10 – Phát hiện bất thường trong log và dữ liệu vận hành mạng  
**Đơn vị:** HKDN UTH – VNPT NET  
**Người phụ trách:** Anh Kiệt (Leader & Architect) – Task K1  
**Ngày cập nhật:** 18/11/2026  

---

## 1. Mục tiêu & Nguyên tắc cốt lõi (Zero Data Leakage)
Khung đánh giá được thiết kế nhằm chuẩn hóa việc chia tập dữ liệu và đo lường hiệu năng của các bộ phát hiện bất thường (**Luật nghiệp vụ, Baseline Z-score, Isolation Forest**), đảm bảo tuyệt đối không xảy ra rò rỉ dữ liệu (**Zero Data Leakage**) với 3 nguyên tắc bất biến:

1. **Bảo toàn trục thời gian & Thứ tự sự kiện:** 
   - Tuyệt đối không xáo trộn ngẫu nhiên (không dùng random shuffle k-fold). Dữ liệu Train luôn xảy ra trước dữ liệu Test.
2. **Khoảng đệm an toàn (Safety Buffer Gap):**
   - Giữa mốc kết thúc của Train và mốc bắt đầu của Test luôn có khoảng đệm tối thiểu `buffer_gap_mins = 60` phút. 
   - Khoảng đệm này đảm bảo không cắt ngang các cửa sổ đặc trưng 5 phút và cách ly các đợt bão log / đợt sự cố giao thoa ranh giới.
3. **Lọc sạch tập huấn luyện (Clean Train Only):**
   - Khi bật cờ `clean_train_only: true`, mọi cửa sổ/dòng log có nhãn bất thường (`y == 1`) đều bị loại bỏ khỏi `train_df`.
   - Các mô hình không giám sát (Isolation Forest, Z-score) chỉ học phân phối trạng thái bình thường thuần túy của hệ thống mạng.
4. **Quy tắc độc lập trạng thái (Reset State):**
   - Ở mỗi fold và mỗi kịch bản thử nghiệm, toàn bộ detector (và bộ trích xuất Drain3) phải được khởi tạo mới hoàn toàn từ đầu thông qua phương thức `.fit(train_df)`.
   - Khóa cập nhật từ điển / ngưỡng khi chạy `.predict(test_df)`.

---

## 2. Cấu hình kịch bản qua YAML (`config/experiment.yaml`)
Hệ thống cho phép tham số hóa toàn bộ chiến lược thực nghiệm không qua can thiệp code:
- **Holdout (theo trục thời gian):**
  - `holdout_80_20`: Train 80% thời gian đầu, Test 20% thời gian cuối.
  - `holdout_75_25`: Train 75% thời gian đầu, Test 25% thời gian cuối.
  - `holdout_70_30`: Train 70% thời gian đầu, Test 30% thời gian cuối.
  - `holdout_60_40`: Train 60% thời gian đầu, Test 40% thời gian cuối.
- **Holdout (theo đợt sự cố - episodes):**
  - `holdout_episodes_70_30`: Cắt theo danh sách các đợt sự cố từ `reports/bgl_episodes.csv`, bảo đảm trọn vẹn toàn bộ một đợt sự cố không bị chia cắt làm đôi giữa hai tập.
- **Cross-Validation chuỗi thời gian:**
  - `expanding_cv`: Cửa sổ Train mở rộng dần theo thời gian (Walk-forward validation), kiểm tra tính thích ứng khi tích lũy thêm dữ liệu.
  - `rolling_cv`: Cửa sổ trượt Train có độ dài cố định trượt tịnh tiến theo thời gian, kiểm tra độ bền vững với sự thay đổi phân phối (concept drift).

---

## 3. Kiến trúc Module Code

### 3.1. `src/evaluation/splitter.py`
Chịu trách nhiệm chia dữ liệu theo đúng hợp đồng `Interface Contract 3`:
- Đầu vào: `DataFrame` dữ liệu cửa sổ đặc trưng (hoặc log chuẩn hóa).
- Đầu ra: Generator sinh các đối tượng `SplitFold` chứa `(fold_idx, train_df, test_df, gap_df, ...)` được nạp sẵn trong bộ nhớ.
- Đã được tích hợp cơ chế bảo vệ ranh giới an toàn và tự động xử lý mốc thời gian dạng epoch UTC.

### 3.2. `src/evaluation/runner.py`
Chịu trách nhiệm điều phối thực nghiệm:
- Đọc cấu hình từ `config/experiment.yaml`.
- Đóng gói giao diện chuẩn `BaseDetector` với 2 hàm bắt buộc:
  ```python
  def fit(self, train_df: pd.DataFrame) -> "BaseDetector": ...
  def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]: ...
  ```
- Nạp 3 detector đại diện: `Rule_FATAL`, `Baseline_ZScore`, `Isolation_Forest`.
- Tính toán đầy đủ các chỉ số: **TP, FP, FN, TN, Precision, Recall, F1-score, Accuracy, MCC**.
- Tổng hợp kết quả đa kịch bản dưới dạng **Mean ± Std** và xuất ra file CSV báo cáo:
  - `reports/evaluation/evaluation_folds_raw.csv`: Chi tiết kết quả từng fold.
  - `reports/evaluation/evaluation_mean_std_summary.csv`: Bảng tổng hợp Mean ± Std cho toàn bộ kịch bản.

---

## 4. Hướng dẫn chạy thực nghiệm

Để chạy toàn bộ khung đánh giá đa kịch bản:
```bash
python src/evaluation/runner.py --config config/experiment.yaml
```

Kết quả sẽ tự động được hiển thị trên console và lưu vào thư mục `reports/evaluation/`.

---

## 5. Bàn giao giao diện cho các thành viên

1. **Cho Duy Khang (Testing độc lập - Task T4):**
   - Viết `tests/test_splitter.py` kiểm tra các điều kiện:
     - `test_df[timestamp].min() - train_df[timestamp].max() >= buffer_gap_seconds`
     - `train_df[target_col].sum() == 0` khi `clean_train_only=True`
     - Các fold không bị trùng lặp hay rò rỉ dữ liệu.
2. **Cho Huy & Phước (Task H7, P1, P3):**
   - Triển khai các detector kế thừa từ class `BaseDetector` trong `src/evaluation/runner.py` để cắm trực tiếp vào vòng lặp chạy tự động.
