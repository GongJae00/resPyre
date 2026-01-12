
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
    RMSE = round(float(np.sqrt(df/m)),2)
    return RMSE


def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    MAE = round(float(df/m),2)
    return MAE

def MAPEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ Computes MAE """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT, normalize=True)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff), axis=1)

    # -- final MAE
    MAPE = round(float((df/m) * 100),2)
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

def calculate_cross_corr_alignment(sig_est, sig_gt):
    """
    Computes the optimal lag to align sig_est to sig_gt using cross-correlation.
    Returns aligned_est, aligned_gt (trimmed to matching length), and the lag.
    """
    if len(sig_est) == 0 or len(sig_gt) == 0:
        return np.array([]), np.array([]), 0
        
    # Standardize inputs to prevent amplitude bias in correlation
    s1 = (sig_est - np.mean(sig_est)) / (np.std(sig_est) + 1e-9)
    s2 = (sig_gt - np.mean(sig_gt)) / (np.std(sig_gt) + 1e-9)
    
    correlation = signal.correlate(s1, s2, mode="full")
    lags = signal.correlation_lags(s1.size, s2.size, mode="full")
    lag = lags[np.argmax(correlation)]
    
    # Apply shift
    # If lag > 0: sig_est is 'ahead' (starts later in time relative to overlap? No.)
    # scipi.signal.correlate(in1, in2): 
    #   if lag is positive, it means in1 shifted by lag matches in2.
    #   So we need to shift in1 by -lag to align? 
    #   Let's use the standard "shift and trim" approach.
    
    if lag > 0:
        # sig_est starts 'lag' samples after sig_gt's start in the best match window
        # We slice sig_est from lag to end, and sig_gt from 0 to matching len
        aligned_est = sig_est[lag:]
        aligned_gt = sig_gt[:len(aligned_est)]
    elif lag < 0:
        # sig_est starts 'lag' samples before. (lag is negative)
        # We slice sig_gt from -lag to end, and sig_est from 0 to matching len
        aligned_gt = sig_gt[-lag:]
        aligned_est = sig_est[:len(aligned_gt)]
    else:
        aligned_est = sig_est
        aligned_gt = sig_gt
        
    # Ensure equal length (sometimes off by 1 due to slicing)
    common_len = min(len(aligned_est), len(aligned_gt))
    return aligned_est[:common_len], aligned_gt[:common_len], lag

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
