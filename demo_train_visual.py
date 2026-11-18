"""
demo_train_visual.py — Ví dụ train MINH HỌA, nhìn thấy từng bước bằng hình.

DỮ LIỆU Ở ĐÂY LÀ GIẢ LẬP (mô phỏng cửa sổ 5 phút giống bgl_mvp_windows.csv),
KHÔNG phải BGL thật. Mục đích: hiểu train gồm những phần nào.

Chạy (Windows, trong thư mục N3.10):
    python demo_train_visual.py
Kết quả: in ra màn hình từng BƯỚC + ảnh trong reports\\demo\\
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # vẽ ra file, không cần cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree

OUT = Path("reports") / "demo"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)          # cố định ngẫu nhiên -> chạy lại ra y hệt

# ---------------------------------------------------------------------------
# BƯỚC 1 — DỮ LIỆU: mỗi dòng = 1 cửa sổ 5 phút, X = đặc trưng, y = nhãn
# ---------------------------------------------------------------------------
N = 3000
t = pd.date_range("2005-06-03", periods=N, freq="5min")
y = (rng.random(N) < 0.04).astype(int)               # ~4% cửa sổ có alert

n_comp = np.where(y == 1, rng.integers(1, 6, N), rng.integers(1, 3, N))
fatal = np.where(y == 1, rng.poisson(25, N), 0)
noise = (y == 0) & (rng.random(N) < 0.05)            # 5% cửa sổ bình thường có FATAL "nhiễu"
fatal = np.where(noise, rng.poisson(6, N) + 1, fatal)
n_comp = np.where(noise, rng.integers(1, 3, N), n_comp)  # nhiễu từ 1–2 component
weak = (y == 1) & (rng.random(N) < 0.25)             # 1/4 alert là alert "nhẹ" (ít FATAL)
fatal = np.where(weak, rng.poisson(4, N) + 1, fatal)

df = pd.DataFrame({"t_start": t, "f_sev_FATAL": fatal,
                   "f_n_components": n_comp, "y": y})
FEAT = ["f_sev_FATAL", "f_n_components"]

print("=" * 70)
print("BƯỚC 1 — DỮ LIỆU (giả lập)")
print(df.head(8).to_string(index=False))
print(f"Tổng {N} cửa sổ, alert = {y.sum()} ({y.mean():.1%})")

# ---------------------------------------------------------------------------
# BƯỚC 2 — CHIA THEO THỜI GIAN (không ngẫu nhiên): 70% đầu học, 30% sau thi
# ---------------------------------------------------------------------------
cut = int(N * 0.7)
train, test = df.iloc[:cut], df.iloc[cut:]
print("\nBƯỚC 2 — CHIA THEO THỜI GIAN")
print(f"train: {train.t_start.min()} -> {train.t_start.max()}  ({len(train)} cửa sổ, {train.y.sum()} alert)")
print(f"test : {test.t_start.min()} -> {test.t_start.max()}  ({len(test)} cửa sổ, {test.y.sum()} alert)")

def scatter(ax, d, title):
    jit = rng.uniform(-0.25, 0.25, len(d))
    n, a = d.y == 0, d.y == 1
    ax.scatter(d.f_n_components[n] + jit[n], d.f_sev_FATAL[n], s=10, c="#9aa5b1", label="bình thường (y=0)")
    ax.scatter(d.f_n_components[a] + jit[a], d.f_sev_FATAL[a], s=22, c="#d64545", label="alert (y=1)")
    ax.set_xlabel("f_n_components"); ax.set_ylabel("f_sev_FATAL"); ax.set_title(title)

fig, ax = plt.subplots(figsize=(7, 5))
scatter(ax, train, "Bước 1–2: dữ liệu train mà mô hình được xem")
ax.axhline(0.5, ls="--", c="k", lw=1); ax.text(4.6, 1.5, "luật FATAL > 0", fontsize=9)
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "1_du_lieu_train.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------------------
# BƯỚC 3 — FIT (TRAIN): xem ranh giới mô hình học được thay đổi khi thêm cây
# ---------------------------------------------------------------------------
print("\nBƯỚC 3 — FIT")
gx, gy = np.meshgrid(np.linspace(0.5, 5.5, 200), np.linspace(-1, 45, 200))
grid = pd.DataFrame({"f_sev_FATAL": gy.ravel(), "f_n_components": gx.ravel()})[FEAT]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, k in zip(axes, [1, 5, 200]):
    m = RandomForestClassifier(n_estimators=k, max_depth=4, class_weight="balanced",
                               random_state=42).fit(train[FEAT], train.y)
    z = m.predict_proba(grid)[:, 1].reshape(gx.shape)
    cs = ax.contourf(gx, gy, z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
    ax.contour(gx, gy, z, levels=[0.5], colors="k", linewidths=1.5)
    scatter(ax, train, f"Sau khi fit với {k} cây")
    print(f"  fit {k:>3} cây: xong")
axes[0].legend(loc="upper left", fontsize=8)
fig.colorbar(cs, ax=axes, label="điểm (score) mô hình cho là alert")
fig.suptitle("Bước 3: mô hình tự vẽ ranh giới (đường đen = ngưỡng 0,5)")
fig.savefig(OUT / "2_fit_ranh_gioi.png", dpi=110, bbox_inches="tight"); plt.close(fig)

model = m                                            # mô hình cuối: 200 cây
fig, ax = plt.subplots(figsize=(13, 6))
plot_tree(model.estimators_[0], feature_names=FEAT, class_names=["bình thường", "alert"],
          filled=True, max_depth=2, fontsize=9, ax=ax, impurity=False, proportion=True)
ax.set_title("Bên trong mô hình: 1 trong 200 cây (2 tầng đầu) — đây là 'luật' máy tự học")
fig.savefig(OUT / "3_mot_cay.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------------------
# BƯỚC 4 — PREDICT trên test (dữ liệu mô hình CHƯA từng thấy)
# ---------------------------------------------------------------------------
test = test.copy()
test["score_rf"] = model.predict_proba(test[FEAT])[:, 1]
test["pred_rf"] = (test.score_rf >= 0.5).astype(int)
test["pred_rule"] = (test.f_sev_FATAL > 0).astype(int)
print("\nBƯỚC 4 — PREDICT (5 cửa sổ test đầu có FATAL > 0)")
print(test[test.f_sev_FATAL > 0].head(5)[["t_start", *FEAT, "score_rf", "pred_rf", "pred_rule", "y"]]
      .to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(test.t_start, test.score_rf, lw=0.8, c="#3b6fb6", label="score RF")
ax.axhline(0.5, ls="--", c="k", lw=1, label="ngưỡng 0,5")
al = test[test.y == 1]
ax.scatter(al.t_start, np.full(len(al), 1.05), marker="v", c="#d64545", s=30, label="alert thật (nhãn)")
ax.set_ylim(-0.05, 1.12); ax.set_ylabel("score"); ax.legend(loc="center right", fontsize=8)
ax.set_title("Bước 4: score của mô hình theo thời gian trên tập test")
fig.tight_layout(); fig.savefig(OUT / "4_score_theo_thoi_gian.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------------------
# BƯỚC 5 — CHẤM ĐIỂM: so luật FATAL với Random Forest
# ---------------------------------------------------------------------------
print("\nBƯỚC 5 — CHẤM ĐIỂM TRÊN TEST")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, name in zip(axes, ["pred_rule", "pred_rf"], ["Luật FATAL > 0", "Random Forest"]):
    cm = confusion_matrix(test.y, test[col], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p, r, f = precision_score(test.y, test[col]), recall_score(test.y, test[col]), f1_score(test.y, test[col])
    print(f"  {name:<16} TP={tp:<3} FP={fp:<3} FN={fn:<3} Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        lab = [["TN", "FP"], ["FN", "TP"]][i][j]
        ax.text(j, i, f"{lab}\n{v}", ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["dự đoán: bt", "dự đoán: alert"]); ax.set_yticks([0, 1], ["thật: bt", "thật: alert"])
    ax.set_title(f"{name}\nP={p:.2f}  R={r:.2f}  F1={f:.2f}")
fig.suptitle("Bước 5: ma trận nhầm lẫn (confusion matrix) trên test")
fig.tight_layout(); fig.savefig(OUT / "5_cham_diem.png", dpi=110); plt.close(fig)

imp = pd.Series(model.feature_importances_, index=FEAT).sort_values(ascending=False)
print("\n  Mức quan trọng đặc trưng (RF):"); print(imp.round(3).to_string())
print(f"\nẢnh đã lưu ở: {OUT.resolve()}")
