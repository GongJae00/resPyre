import numpy as np
import warnings

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None  # Optional dependency used only for interactive plots

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
        elif m == 'PCC':
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
    MAX = df
    return MAX


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


def printErrors(RMSE, MAE, MAX, PCC, CCC):
    print("\n    * Errors: RMSE = %.2f, MAE = %.2f, MAX = %.2f, PCC = %.2f, CCC = %.2f" %
          (RMSE, MAE, MAX, PCC, CCC))


def displayErrors(bpmES, bpmGT, timesES=None, timesGT=None):
    """"Plots errors"""
    if go is None:
        raise ImportError(
            "Plotly is required for displayErrors(), but the 'plotly' package is not installed."
        )
    if type(bpmES) == list:
        bpmES = np.expand_dims(bpmES, axis=0)
    if type(bpmES) == np.ndarray:
        if len(bpmES.shape) == 1:
            bpmES = np.expand_dims(bpmES, axis=0)


    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n, m = diff.shape  # n = num channels, m = bpm length
    df = np.abs(diff)
    dfMean = np.around(np.mean(df, axis=1), 1)

    # -- plot errors
    fig = go.Figure()
    name = 'Ch 1 (µ = ' + str(dfMean[0]) + ' )'
    fig.add_trace(go.Scatter(
        x=timesES, y=df[0, :], name=name, mode='lines+markers'))
    if n > 1:
        name = 'Ch 2 (µ = ' + str(dfMean[1]) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=df[1, :], name=name, mode='lines+markers'))
        name = 'Ch 3 (µ = ' + str(dfMean[2]) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=df[2, :], name=name, mode='lines+markers'))
    fig.update_layout(xaxis_title='Times (sec)',
                      yaxis_title='MAE', showlegend=True)
    fig.show()

    # -- plot bpm Gt and ES
    fig = go.Figure()
    GTmean = np.around(np.mean(bpmGT), 1)
    name = 'GT (µ = ' + str(GTmean) + ' )'
    fig.add_trace(go.Scatter(x=timesGT, y=bpmGT,
                             name=name, mode='lines+markers'))
    ESmean = np.around(np.mean(bpmES[0, :]), 1)
    name = 'ES1 (µ = ' + str(ESmean) + ' )'
    fig.add_trace(go.Scatter(
        x=timesES, y=bpmES[0, :], name=name, mode='lines+markers'))
    if n > 1:
        ESmean = np.around(np.mean(bpmES[1, :]), 1)
        name = 'ES2 (µ = ' + str(ESmean) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=bpmES[1, :], name=name, mode='lines+markers'))
        ESmean = np.around(np.mean(bpmES[2, :]), 1)
        name = 'E3 (µ = ' + str(ESmean) + ' )'
        fig.add_trace(go.Scatter(
            x=timesES, y=bpmES[2, :], name=name, mode='lines+markers'))

    fig.update_layout(xaxis_title='Times (sec)',
                      yaxis_title='BPM', showlegend=True)
    fig.show()


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
