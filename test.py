import os
import glob
import pickle
import numpy as np
import pandas as pd

# ==========================
# 설정
# ==========================
data_dir = "/home/gongjae/VisualStudioCode/resPyre/results/cohface_motion_oscillator/data"
FMIN = 0.08  # Hz
FMAX = 0.5   # Hz

# ==========================
# 유틸 함수들
# ==========================
def align_to_gt(gt, fs_gt, est, fs_est):
    """
    gt와 est를 시간축 기준으로 정렬:
    - t_gt와 t_est의 공통 구간까지만 사용
    - est를 t_gt에 선형보간해서 샘플링
    반환: gt_aligned, est_aligned (같은 길이), fs_gt
    """
    gt = np.asarray(gt, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)

    if len(gt) == 0 or len(est) == 0:
        return None, None, None

    t_gt = np.arange(len(gt)) / fs_gt
    t_est = np.arange(len(est)) / fs_est

    t_max = min(t_gt[-1], t_est[-1])
    mask = t_gt <= t_max
    if not np.any(mask):
        return None, None, None

    t_gt_short = t_gt[mask]
    gt_short = gt[mask]

    est_interp = np.interp(t_gt_short, t_est, est)

    return gt_short, est_interp, fs_gt


def fft_bandpass(x, fs, fmin, fmax):
    """
    FFT 기반 ideal bandpass: [fmin, fmax] 외의 성분을 0으로
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    band = (freqs >= fmin) & (freqs <= fmax)
    X_filtered = np.zeros_like(X)
    X_filtered[band] = X[band]

    y = np.fft.irfft(X_filtered, n=len(x))
    return y


def pearsonr_np(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return np.nan

    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.sqrt(np.sum(x_c ** 2) * np.sum(y_c ** 2))
    if denom == 0:
        return np.nan
    return np.sum(x_c * y_c) / denom


def ccc_np(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return np.nan

    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    r = pearsonr_np(x, y)
    if np.isnan(r):
        return np.nan

    denom = var_x + var_y + (mu_x - mu_y) ** 2
    if denom == 0:
        return np.nan

    return 2 * r * np.sqrt(var_x * var_y) / denom


def dominant_frequency(x, fs, fmin, fmax):
    """
    주어진 신호 x에서 [fmin, fmax] 대역 내에서
    최대 스펙트럼 성분의 주파수(Hz)를 반환.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.nan

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return np.nan

    mag = np.abs(X[band])
    if mag.size == 0:
        return np.nan

    idx = np.argmax(mag)
    return freqs[band][idx]  # Hz


# ==========================
# 메인 루프: 모든 .pkl 처리
# ==========================
pkl_paths = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))

rows = []  # trial × method별 metrics 저장

for path in pkl_paths:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] {path} 로드 실패: {e}")
        continue

    if not all(k in data for k in ("gt", "estimates", "fps", "fs_gt")):
        print(f"[WARN] {path} : 필요한 키(gt, estimates, fps, fs_gt) 없음 → 건너뜀")
        continue

    trial_key = os.path.splitext(os.path.basename(path))[0]
    gt = np.asarray(data["gt"], dtype=np.float64)
    fs_gt = float(data["fs_gt"])
    fs_est = float(data["fps"])
    estimates = data["estimates"]  # [{method, estimate}, ...]

    for est in estimates:
        method_name = est.get("method", "unknown")
        est_signal = np.asarray(est.get("estimate"), dtype=np.float64)

        # 1) 시간축 정렬 (gt 기준)
        gt_aligned, est_aligned, fs_common = align_to_gt(gt, fs_gt, est_signal, fs_est)

        if gt_aligned is None or est_aligned is None:
            mae_raw = np.nan
            pcc_band = np.nan
            ccc_band = np.nan
            pfreq_gt_bpm = np.nan
            pfreq_est_bpm = np.nan
            pfreq_error_bpm = np.nan
            pfreq_abs_error_bpm = np.nan
        else:
            # 2) Raw MAE (정렬된 시계열 기준)
            mae_raw = float(np.mean(np.abs(gt_aligned - est_aligned)))

            # 3) 0.08–0.5 Hz bandpass
            gt_filt = fft_bandpass(gt_aligned, fs_common, FMIN, FMAX)
            est_filt = fft_bandpass(est_aligned, fs_common, FMIN, FMAX)

            # 4) PCC / CCC (bandpass 신호)
            pcc_band = float(pearsonr_np(gt_filt, est_filt))
            ccc_band = float(ccc_np(gt_filt, est_filt))

            # 5) Predominant frequency (Hz → BPM)
            pfreq_gt_hz = float(dominant_frequency(gt_aligned, fs_common, FMIN, FMAX))
            pfreq_est_hz = float(dominant_frequency(est_aligned, fs_common, FMIN, FMAX))

            if np.isnan(pfreq_gt_hz) or np.isnan(pfreq_est_hz):
                pfreq_gt_bpm = np.nan
                pfreq_est_bpm = np.nan
                pfreq_error_bpm = np.nan
                pfreq_abs_error_bpm = np.nan
            else:
                pfreq_gt_bpm = pfreq_gt_hz * 60.0
                pfreq_est_bpm = pfreq_est_hz * 60.0
                pfreq_error_bpm = pfreq_est_bpm - pfreq_gt_bpm
                pfreq_abs_error_bpm = abs(pfreq_error_bpm)

        rows.append({
            "trial": trial_key,
            "method": method_name,
            "mae_raw": mae_raw,
            "pcc_band": pcc_band,
            "ccc_band": ccc_band,
            "pfreq_gt_bpm": pfreq_gt_bpm,
            "pfreq_est_bpm": pfreq_est_bpm,
            "pfreq_error_bpm": pfreq_error_bpm,
            "pfreq_abs_error_bpm": pfreq_abs_error_bpm,
        })


# ==========================
# 1) trial × method별 메트릭 CSV
# ==========================
df_tm = pd.DataFrame(rows)
df_tm.sort_values(["method", "trial"], inplace=True)

csv_trial_method = os.path.join(data_dir, "metrics_trial_method.csv")
df_tm.to_csv(csv_trial_method, index=False)
print(f"[INFO] Trial × Method metrics 저장 완료: {csv_trial_method}")


# ==========================
# 2) method별 통계 (MAE, PCC, CCC, PredFreq(BPM) MAE/RMSE)
# ==========================
method_stats_rows = []

for method_name, g in df_tm.groupby("method"):
    mae_vals = g["mae_raw"].dropna().values
    pcc_vals = g["pcc_band"].dropna().values
    ccc_vals = g["ccc_band"].dropna().values
    pf_err_bpm = g["pfreq_error_bpm"].dropna().values

    def stat(v):
        if v.size == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        return (v.size, np.mean(v), np.median(v), np.std(v), np.min(v), np.max(v))

    mae_count, mae_mean, mae_median, mae_std, mae_min, mae_max = stat(mae_vals)
    pcc_count, pcc_mean, pcc_median, pcc_std, pcc_min, pcc_max = stat(pcc_vals)
    ccc_count, ccc_mean, ccc_median, ccc_std, ccc_min, ccc_max = stat(ccc_vals)

    # Predominant frequency error in BPM: MAE, RMSE
    if pf_err_bpm.size == 0:
        pfreq_mae_bpm = np.nan
        pfreq_rmse_bpm = np.nan
    else:
        pfreq_mae_bpm = float(np.mean(np.abs(pf_err_bpm)))          # MAE [BPM]
        pfreq_rmse_bpm = float(np.sqrt(np.mean(pf_err_bpm ** 2)))   # RMSE [BPM]

    method_stats_rows.append({
        "method": method_name,

        "mae_count": mae_count,
        "mae_mean": mae_mean,
        "mae_median": mae_median,
        "mae_std": mae_std,
        "mae_min": mae_min,
        "mae_max": mae_max,

        "pcc_mean": pcc_mean,
        "pcc_median": pcc_median,
        "pcc_std": pcc_std,
        "pcc_min": pcc_min,
        "pcc_max": pcc_max,

        "ccc_mean": ccc_mean,
        "ccc_median": ccc_median,
        "ccc_std": ccc_std,
        "ccc_min": ccc_min,
        "ccc_max": ccc_max,

        "pfreq_mae_bpm": pfreq_mae_bpm,   # predominant frequency MAE [BPM]
        "pfreq_rmse_bpm": pfreq_rmse_bpm, # predominant frequency RMSE [BPM]
    })

df_ms = pd.DataFrame(method_stats_rows)
df_ms.sort_values("method", inplace=True)

csv_method_stats = os.path.join(data_dir, "metrics_method_stats.csv")
df_ms.to_csv(csv_method_stats, index=False)
print(f"[INFO] Method별 metrics 통계 저장 완료: {csv_method_stats}")
