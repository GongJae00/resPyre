
import numpy as np
import warnings
from scipy import signal
from scipy.stats import entropy

def getErrors(bpmES, bpmGT, timesES, timesGT, metrics):
    """ Computes various error/quality measures (multiple time windows case)"""
    if type(bpmES) == list:
        bpmES = np.expand_dims(bpmES, axis=0)
    if type(bpmES) == np.ndarray:
        if len(bpmES.shape) == 1:
            bpmES = np.expand_dims(bpmES, axis=0)
    err = []
    for m in metrics:
        if m == 'RMSE':
            e = RMSEerror(bpmES, bpmGT, timesES, timesGT)
        elif m == 'MAE':
            e = MAEerror(bpmES, bpmGT, timesES, timesGT)
        elif m == 'MAPE':
            e = MAPEerror(bpmES, bpmGT, timesES, timesGT)
        elif m == 'MAX':
            e = MAXError(bpmES, bpmGT, timesES, timesGT)
        elif m in ('PCC', 'R', 'PearsonR'):
            e = PearsonCorr(bpmES, bpmGT, timesES, timesGT)
        elif m == 'CCC':
            e = LinCorr(bpmES, bpmGT, timesES, timesGT)
        err.append(e)
    err.append([bpmES, bpmGT])
    return err


def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes RMSE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c, j], 2)

    # -- final RMSE
    rmse_arr = np.sqrt(df / max(m, 1))
    rmse_val = rmse_arr[0] if rmse_arr.size else np.nan
    RMSE = round(float(rmse_val), 2) if np.isfinite(rmse_val) else float('nan')
    return RMSE


def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    mae_arr = df / max(m, 1)
    mae_val = mae_arr[0] if mae_arr.size else np.nan
    MAE = round(float(mae_val), 2) if np.isfinite(mae_val) else float('nan')
    return MAE

def MAPEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT, normalize=True)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    mape_arr = (df / max(m, 1)) * 100.0
    mape_val = mape_arr[0] if mape_arr.size else np.nan
    MAPE = round(float(mape_val), 2) if np.isfinite(mape_val) else float('nan')
    return MAPE


def MAXError(bpmES, bpmGT, timesES=None, timesGT=None):
    """ computes MAX """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.max(np.abs(diff), axis=1)

    # -- final MAX
    result = df[0] if df.size else np.nan
    return round(float(result), 2)


def PearsonCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    """Computes Pearson correlation coefficient with guards against degenerate input."""
    from scipy import stats

    bpmES_arr = np.asarray(bpmES, dtype=np.float64)
    if bpmES_arr.ndim == 1:
        bpmES_arr = np.expand_dims(bpmES_arr, axis=0)
    bpmGT_arr = np.asarray(bpmGT, dtype=np.float64).reshape(-1)

    diff = bpm_diff(bpmES_arr, bpmGT_arr, timesES, timesGT)
    n, m = diff.shape
    if m < 2:
        return float('nan')

    CC = np.full(n, np.nan, dtype=np.float64)
    eps = 1e-6
    for c in range(n):
        x = diff[c, :] + bpmES_arr[c, :]
        y = bpmES_arr[c, :]
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(finite_mask) < 2:
            continue
        x_valid = x[finite_mask]
        y_valid = y[finite_mask]
        sx = np.std(x_valid, dtype=np.float64)
        sy = np.std(y_valid, dtype=np.float64)
        if (not np.isfinite(sx)) or (sx < eps) or (not np.isfinite(sy)) or (sy < eps):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            try:
                r, _ = stats.pearsonr(x_valid, y_valid)
            except Exception:
                r = np.nan
        CC[c] = r if np.isfinite(r) else np.nan

    result = CC[0] if CC.size else np.nan
    if not np.isfinite(result):
        return float('nan')
    return round(float(result), 2)


def LinCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    """Computes Lin's Concordance Correlation Coefficient with degenerate-input guards."""
    bpmES_arr = np.asarray(bpmES, dtype=np.float64)
    if bpmES_arr.ndim == 1:
        bpmES_arr = np.expand_dims(bpmES_arr, axis=0)
    bpmGT_arr = np.asarray(bpmGT, dtype=np.float64).reshape(-1)

    diff = bpm_diff(bpmES_arr, bpmGT_arr, timesES, timesGT)
    n, m = diff.shape
    if m < 2:
        return float('nan')

    CCC = np.full(n, np.nan, dtype=np.float64)
    eps = 1e-6
    for c in range(n):
        x = bpmES_arr[c, :]
        y = diff[c, :] + bpmES_arr[c, :]
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(finite_mask) < 2:
            continue
        x_valid = x[finite_mask]
        y_valid = y[finite_mask]
        sx = np.std(x_valid, dtype=np.float64)
        sy = np.std(y_valid, dtype=np.float64)
        if (not np.isfinite(sx)) or (sx < eps) or (not np.isfinite(sy)) or (sy < eps):
            continue
        ccc = concordance_correlation_coefficient(x_valid, y_valid)
        CCC[c] = ccc if np.isfinite(ccc) else np.nan

    result = CCC[0] if CCC.size else np.nan
    if not np.isfinite(result):
        return float('nan')
    return round(float(result), 2)


def printErrors(RMSE, MAE, MAX, R, CCC=None):
    if CCC is None:
        print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, R = %.2f" %
              (RMSE, MAE, MAX, R))
    else:
        print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, R = %.2f, CCC = %.2f" %
              (RMSE, MAE, MAX, R, CCC))


def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None, normalize=False):
    n, m = bpmES.shape  # n = num channels, m = bpm length

    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES

    diff = np.zeros((n, m))
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            if not normalize:
                diff[c, j] = bpmGT[i]-bpmES[c, j]
            else:
                diff[c, j] = (bpmGT[i]-bpmES[c, j]) / bpmGT[i]
    return diff

def concordance_correlation_coefficient(series_a, series_b):
    """Compute Lin's concordance correlation coefficient with robust guards."""
    arr_a = np.asarray(series_a, dtype=np.float64)
    arr_b = np.asarray(series_b, dtype=np.float64)
    finite_mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    if np.count_nonzero(finite_mask) < 2:
        return np.nan
    a = arr_a[finite_mask]
    b = arr_b[finite_mask]
    eps = 1e-6
    std_a = np.std(a, dtype=np.float64)
    std_b = np.std(b, dtype=np.float64)
    if (not np.isfinite(std_a)) or (std_a < eps) or (not np.isfinite(std_b)) or (std_b < eps):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        corr_matrix = np.corrcoef(a, b)
    if corr_matrix.shape[0] < 2 or not np.isfinite(corr_matrix[0, 1]):
        return np.nan
    rho = corr_matrix[0, 1]
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a)
    var_b = np.var(b)
    numerator = 2 * rho * std_a * std_b
    denominator = var_a + var_b + (mean_a - mean_b) ** 2
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator

def calculate_cross_corr_alignment(sig_est, sig_gt, fs_est=None, fs_gt=None):
    """
    Computes the optimal lag to align sig_est to sig_gt using cross-correlation.
    If fs_est and fs_gt are provided and different, sig_est is resampled to fs_gt.
    Returns aligned_est, aligned_gt (trimmed to matching length), and the lag in seconds.
    """
    if len(sig_est) == 0 or len(sig_gt) == 0:
        return np.array([]), np.array([]), 0.0
        
    # Handle resampling if frequencies are different
    if fs_est is not None and fs_gt is not None and abs(fs_est - fs_gt) > 1e-6:
        # Resample sig_est to fs_gt using linear interpolation to avoid edge artifacts (ringing)
        t_orig = np.arange(len(sig_est)) / fs_est
        num_samples_target = int(round(len(sig_est) * fs_gt / fs_est))
        t_new = np.arange(num_samples_target) / fs_gt
        s1_raw = np.interp(t_new, t_orig, sig_est.flatten())
        effective_fs = fs_gt
    else:
        s1_raw = sig_est.flatten()
        effective_fs = fs_gt or fs_est or 1.0 # Fallback

    s2_raw = sig_gt.flatten()
    
    # Standardize inputs to prevent amplitude bias in correlation
    s1 = (s1_raw - np.mean(s1_raw)) / (np.std(s1_raw) + 1e-9)
    s2 = (s2_raw - np.mean(s2_raw)) / (np.std(s2_raw) + 1e-9)
    
    correlation = signal.correlate(s1, s2, mode="full")
    lags = signal.correlation_lags(s1.size, s2.size, mode="full")
    lag_idx = lags[np.argmax(correlation)]
    
    # Apply shift
    if lag_idx > 0:
        # s1 starts 'lag_idx' samples after s2's start in the best match window
        aligned_est = s1_raw[lag_idx:]
        aligned_gt = s2_raw[:len(aligned_est)]
    elif lag_idx < 0:
        # s1 starts 'lag_idx' samples before. (lag is negative)
        aligned_gt = s2_raw[-lag_idx:]
        aligned_est = s1_raw[:len(aligned_gt)]
    else:
        aligned_est = s1_raw
        aligned_gt = s2_raw
        
    # Ensure equal length
    common_len = min(len(aligned_est), len(aligned_gt))
    lag_sec = lag_idx / effective_fs
    
    return aligned_est[:common_len], aligned_gt[:common_len], lag_sec

def bland_altman_stats(est, gt):
    """
    Computes Bland-Altman statistics: Bias and Limits of Agreement (LoA).
    """
    est = np.asarray(est).flatten()
    gt = np.asarray(gt).flatten()
    
    # Filter NaNs
    mask = np.isfinite(est) & np.isfinite(gt)
    est = est[mask]
    gt = gt[mask]
    
    if len(est) < 2:
        return np.nan, np.nan, np.nan
        
    diffs = est - gt
    bias = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd
    
    return bias, lower_loa, upper_loa

def calculate_spectral_snr(sig, fs, min_hz=0.08, max_hz=0.5):
    """
    Computes Spectral SNR using De Haan's method (Fundamental + 1st Harmonic).
    Signal is expected to be windowed already.
    """
    # Use padding for better freq resolution
    n = len(sig)
    nfft = max(2048, int(2**np.ceil(np.log2(n))))
    
    freqs, psd = signal.periodogram(sig, fs, window='hamming', nfft=nfft)
    
    # Limit to physiological range
    mask = (freqs >= min_hz) & (freqs <= 4.0) # Up to 4Hz for noise floor
    freqs = freqs[mask]
    psd = psd[mask]
    
    if len(psd) == 0:
        return np.nan
        
    # Find peak in the respiratory band
    resp_mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(resp_mask):
        return np.nan
        
    resp_psd = psd[resp_mask]
    resp_freqs = freqs[resp_mask]
    
    peak_idx = np.argmax(resp_psd)
    f_peak = resp_freqs[peak_idx]
    
    # Define signal bands (Fundamental + 1st Harmonic)
    # De Haan uses +/- 3 BPM (0.05 Hz) -> Window width 0.1 Hz? 
    # Let's use +/- 0.05 Hz around peak
    half_width = 0.05
    
    f1_mask = (freqs >= f_peak - half_width) & (freqs <= f_peak + half_width)
    f2_mask = (freqs >= 2*f_peak - half_width) & (freqs <= 2*f_peak + half_width)
    
    signal_mask = f1_mask | f2_mask
    noise_mask = ~signal_mask
    
    power_signal = np.sum(psd[signal_mask])
    power_noise = np.sum(psd[noise_mask])
    
    if power_noise == 0:
        return np.inf
        
    snr = 10 * np.log10(power_signal / power_noise)
    return snr

def calculate_breathing_dynamics(sig, fs):
    """
    Computes physiological breathing dynamics:
    1. Mean I:E Ratio (Inhalation time / Exhalation time)
    2. Mean PPI (Peak-to-Peak Interval in seconds)
    
    Returns: ie_ratio (float), ppi_mean (float), peaks (indices)
    """
    # Simple peak/trough detection on filtered signal
    # We assume signal is somewhat clean (bandpassed).
    # Inhalation = Trough to Peak? Or Peak to Trough?
    # Usually rPPG: Inspiration -> decrease in PPG intensity (valleys)? 
    # Or depends on sign. Let's assume standard respiratory signal convention:
    # Inspiration = Rising edge (Trough -> Peak)
    # Expiration = Falling edge (Peak -> Trough)
    # This is arbitrary without knowing sensor polarity, but consistency matches.
    
    # Use prominence to avoid noise
    peaks, _ = signal.find_peaks(sig, prominence=0.3*np.std(sig), distance=int(fs*1.0))
    troughs, _ = signal.find_peaks(-sig, prominence=0.3*np.std(sig), distance=int(fs*1.0))
    
    if len(peaks) < 2 or len(troughs) < 2:
        return np.nan, np.nan, peaks
        
    s_peaks = np.sort(peaks)
    
    # PPI
    ppis = np.diff(s_peaks) / fs
    mean_ppi = np.mean(ppis)
    
    # I:E Ratio
    # Naive approach: Find nearest trough before each peak (Inspiration) 
    # and nearest trough after each peak (Expiration).
    
    ie_ratios = []
    # Sort all events
    events = [(t, 't') for t in troughs] + [(p, 'p') for p in peaks]
    events.sort(key=lambda x: x[0])
    
    # Iterate to find Trough -> Peak -> Trough patterns
    for i in range(len(events)-2):
        e1, type1 = events[i]
        e2, type2 = events[i+1]
        e3, type3 = events[i+2]
        
        if type1 == 't' and type2 == 'p' and type3 == 't':
            t_in = (e2 - e1) / fs
            t_ex = (e3 - e2) / fs
            if t_ex > 0:
                ie_ratios.append(t_in / t_ex)
                
    mean_ie = np.mean(ie_ratios) if ie_ratios else np.nan
    
    return mean_ie, mean_ppi, peaks

def calculate_spectral_shape_metrics(est_psd, gt_psd):
    """
    Computes spectral shape similarity metrics:
    1. KL Divergence (Probabilistic difference)
    2. Spectral Entropy Difference (Complexity difference)
    
    Input PSDs should be computed on the same frequency grid.
    """
    # Normalize to create probability distributions (sum = 1)
    p = np.asarray(est_psd) + 1e-12
    q = np.asarray(gt_psd) + 1e-12
    
    p_norm = p / np.sum(p)
    q_norm = q / np.sum(q)
    
    # KL Divergence: D_KL(P || Q) = sum(p * log(p/q))
    # We measure how much Information corresponds to GT (Q) is lost when using Est (P)?
    # Usually D_KL(GT || Est) or D_KL(Est || GT). 
    # Let's use symmetric Jensen-Shannon or just standard KL(GT || Est) 
    # "How well does Est approximate GT?" -> D_KL(GT || Est)
    kl_div = entropy(q_norm, p_norm)
    
    # Spectral Entropy
    # Measures how 'flat' or 'peaky' the spectrum is.
    # High entropy = White noise (flat). Low entropy = Pure sine (peaky).
    est_ent = entropy(p_norm)
    gt_ent = entropy(q_norm)
    abs_ent_err = abs(est_ent - gt_ent)
    
    return kl_div, abs_ent_err, est_ent

def calculate_dtw_distance(s1, s2):
    """
    Computes the Dynamic Time Warping (DTW) distance between two signals using FastDTW algorithm (approximation).
    If fastdtw depends not installed, falls back to a simple numpy implementation (slow for long sequences).
    """
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        
        # Ensure 1D and float
        x = np.array(s1).reshape(-1, 1)
        y = np.array(s2).reshape(-1, 1)
        
        distance, path = fastdtw(x, y, dist=euclidean)
        # Normalize by path length to make independent of duration?
        # Usually DTW distance is sum. To compare across videos of diff lengths, we should normalize.
        # Length of path is roughly proportional to len(s1).
        norm_dist = distance / len(path)
        return norm_dist
        
    except ImportError:
        # Fallback: Basic DTW implementation
        # Warning: O(N*M) complexity. Only use for short aligned windows if possible.
        n, m = len(s1), len(s2)
        # If too long, downsample
        if n > 100:
            target_len = 100
            s1 = signal.resample(s1, target_len)
            s2 = signal.resample(s2, target_len)
            n, m = len(s1), len(s2)
            
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, 1:] = np.inf
        dtw_matrix[1:, 0] = np.inf
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(s1[i-1] - s2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                              dtw_matrix[i, j-1],    # deletion
                                              dtw_matrix[i-1, j-1])  # match
        
        distance = dtw_matrix[n, m]
        # Normalize approximated path length (max(N, M))
        return distance / max(n, m)


# ──────────────────────────────────────────────────────────────────────
# Bayesian Filter Calibration Metrics (Phase 7)
# ──────────────────────────────────────────────────────────────────────

def nis_calibration_chi2(nis_sequence, dof=1, alpha=0.05):
    """
    NIS calibration test (Spec §11.2).

    Tests whether the Normalized Innovation Squared (NIS) sequence is
    consistent with a χ²(dof) distribution. A well-calibrated filter
    should produce NIS ~ χ²(1) for scalar observations.

    Parameters
    ----------
    nis_sequence : array-like
        NIS values from the filter (one per frame).
    dof : int
        Observation dimension (default 1 for scalar).
    alpha : float
        Significance level for the two-sided test.

    Returns
    -------
    dict with:
        mean_nis : float — should be close to `dof`
        pass_chi2 : bool — True if mean NIS is within the acceptance band
        pval : float — p-value from scipy.stats.chi2 test
        ci_lower, ci_upper : float — 95% acceptance bounds
    """
    nis = np.asarray(nis_sequence, dtype=np.float64)
    nis = nis[np.isfinite(nis)]
    n = len(nis)
    if n < 10:
        return {'mean_nis': float('nan'), 'pass_chi2': False,
                'pval': float('nan'), 'ci_lower': float('nan'),
                'ci_upper': float('nan'), 'n_valid': 0}

    from scipy.stats import chi2
    mean_nis = float(np.mean(nis))
    # Under H0: n * mean_nis ~ χ²(n * dof)
    # Acceptance interval: [χ²_{α/2}(n·dof)/n, χ²_{1-α/2}(n·dof)/n]
    df = n * dof
    ci_lower = float(chi2.ppf(alpha / 2, df) / n)
    ci_upper = float(chi2.ppf(1 - alpha / 2, df) / n)
    pass_chi2 = ci_lower <= mean_nis <= ci_upper

    # Also compute two-sided p-value with log-space tails to avoid
    # floating-point underflow to exact zero for extreme statistics.
    x = n * mean_nis
    log_cdf = float(chi2.logcdf(x, df))
    log_sf = float(chi2.logsf(x, df))
    log_tail = min(log_cdf, log_sf)  # smaller tail drives two-sided p-value

    min_pos = float(np.nextafter(0.0, 1.0))  # smallest positive float
    if np.isfinite(log_tail):
        pval = float(np.exp(np.log(2.0) + log_tail))
        pval = float(np.clip(pval, min_pos, 1.0))
    else:
        pval = min_pos

    return {
        'mean_nis': mean_nis,
        'pass_chi2': pass_chi2,
        'pval': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_valid': n,
    }


def coverage_percentage(freq_track, freq_gt, sigma_freq, k=2.0):
    """
    Coverage percentage (Spec §11.3).

    Percentage of ground truth values falling within the filter's
    k·σ credible interval around the estimated frequency.

    Parameters
    ----------
    freq_track : array-like
        Estimated frequencies (Hz) from filter.
    freq_gt : array-like
        Ground truth frequencies (Hz).
    sigma_freq : array-like
        Standard deviation of frequency estimate (Hz).
    k : float
        Number of standard deviations for the interval (default 2 = ~95%).

    Returns
    -------
    dict with:
        coverage : float — percentage [0, 100]
        n_inside : int — number of GT frames within the interval
        n_total : int — total valid frames
    """
    ft = np.asarray(freq_track, dtype=np.float64)
    fg = np.asarray(freq_gt, dtype=np.float64)
    sf = np.asarray(sigma_freq, dtype=np.float64)

    # Align lengths
    n = min(len(ft), len(fg), len(sf))
    ft, fg, sf = ft[:n], fg[:n], sf[:n]

    # Filter to valid entries
    valid = np.isfinite(ft) & np.isfinite(fg) & np.isfinite(sf) & (sf > 0)
    ft_v, fg_v, sf_v = ft[valid], fg[valid], sf[valid]
    n_total = int(np.sum(valid))

    if n_total == 0:
        return {'coverage': 0.0, 'n_inside': 0, 'n_total': 0}

    inside = np.abs(fg_v - ft_v) <= k * sf_v
    n_inside = int(np.sum(inside))
    coverage = 100.0 * n_inside / n_total

    return {
        'coverage': float(coverage),
        'n_inside': n_inside,
        'n_total': n_total,
    }


def stability_duration(freq_track, fs, eps_hz=0.02):
    """
    Stability duration (Spec §11.4).

    Longest contiguous window where |Δf| < ε_hz (frame-to-frame frequency
    change is below threshold). Reported in seconds.

    Parameters
    ----------
    freq_track : array-like
        Estimated frequencies (Hz) from filter.
    fs : float
        Sampling rate (frames per second).
    eps_hz : float
        Frequency stability threshold in Hz.

    Returns
    -------
    dict with:
        max_stable_sec : float — longest stable window in seconds
        max_stable_frames : int — longest stable window in frames
        total_stable_pct : float — percentage of stable frames
    """
    ft = np.asarray(freq_track, dtype=np.float64)
    ft = ft[np.isfinite(ft)]
    n = len(ft)

    if n < 2:
        return {'max_stable_sec': 0.0, 'max_stable_frames': 0,
                'total_stable_pct': 0.0}

    diffs = np.abs(np.diff(ft))
    stable = diffs < eps_hz

    # Find longest run of True
    max_run = 0
    current_run = 0
    for s in stable:
        if s:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    # +1 because diff reduces length by 1
    max_frames = max_run + 1 if max_run > 0 else 0
    total_stable = int(np.sum(stable))
    total_pct = 100.0 * total_stable / len(stable) if len(stable) > 0 else 0.0

    return {
        'max_stable_sec': float(max_frames / fs),
        'max_stable_frames': max_frames,
        'total_stable_pct': float(total_pct),
    }


# ──────────────────────────────────────────────────────────────────────
# Waveform-level metrics for PARH-OSSM dual-output evaluation
# ──────────────────────────────────────────────────────────────────────

def waveform_ccc(sig_est, sig_gt):
    """CCC between estimated and ground-truth waveforms (sample-level).

    This is for z_full evaluation, NOT for rate BPM sequences.
    Signals should be at the same sampling rate and aligned.

    Returns float CCC in [-1, 1] or NaN.
    """
    a = np.asarray(sig_est, dtype=np.float64).flatten()
    b = np.asarray(sig_gt, dtype=np.float64).flatten()
    n = min(len(a), len(b))
    if n < 2:
        return float('nan')
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return float('nan')
    return float(concordance_correlation_coefficient(a[mask], b[mask]))


def waveform_mae(sig_est, sig_gt):
    """Sample-level MAE between waveforms.

    For z_full evaluation. Both signals should be normalised
    (e.g. robust z-scored) before comparison.
    """
    a = np.asarray(sig_est, dtype=np.float64).flatten()
    b = np.asarray(sig_gt, dtype=np.float64).flatten()
    n = min(len(a), len(b))
    if n < 1:
        return float('nan')
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 1:
        return float('nan')
    return float(np.mean(np.abs(a[mask] - b[mask])))


def waveform_dtw(sig_est, sig_gt):
    """DTW distance between waveforms. Wrapper around calculate_dtw_distance.

    For z_full evaluation. Returns normalised DTW distance or NaN.
    """
    a = np.asarray(sig_est, dtype=np.float64).flatten()
    b = np.asarray(sig_gt, dtype=np.float64).flatten()
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    mask_a = np.isfinite(a)
    mask_b = np.isfinite(b)
    if np.count_nonzero(mask_a) < 2 or np.count_nonzero(mask_b) < 2:
        return float('nan')
    try:
        return float(calculate_dtw_distance(a[mask_a], b[mask_b]))
    except Exception:
        return float('nan')


def compute_dual_output_metrics(result_dict, gt_signal, fs):
    """Compute rate metrics (z_osc) and waveform metrics (z_full) from PARH-OSSM result.

    Args:
        result_dict: dict from PARH-OSSM head containing z_osc, z_full, track_hz, etc.
        gt_signal: ground-truth respiratory waveform (same fs)
        fs: sampling rate

    Returns:
        dict with keys:
            rate_metrics: {MAE, RMSE, PearsonR} (from z_osc → track_hz → BPM)
            waveform_metrics: {CCC, wMAE, DTW} (from z_full vs gt_signal)
            output_type: 'dual' or 'single'
    """
    metrics_out = {"output_type": "single", "rate_metrics": {}, "waveform_metrics": {}}

    z_full = result_dict.get("z_full")
    if z_full is not None and gt_signal is not None:
        gt = np.asarray(gt_signal, dtype=np.float64).flatten()
        zf = np.asarray(z_full, dtype=np.float64).flatten()

        # Align via cross-correlation
        aligned_est, aligned_gt, _ = calculate_cross_corr_alignment(zf, gt, fs, fs)

        if len(aligned_est) > 1:
            metrics_out["waveform_metrics"] = {
                "CCC": waveform_ccc(aligned_est, aligned_gt),
                "wMAE": waveform_mae(aligned_est, aligned_gt),
                "DTW": waveform_dtw(aligned_est, aligned_gt),
            }
            metrics_out["output_type"] = "dual"

    return metrics_out
