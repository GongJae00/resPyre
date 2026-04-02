import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy import signal
from scipy.signal import butter, filtfilt
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, x): self.x = x
        def __iter__(self): return iter(self.x)
        format_dict = {"elapsed": 0.0}
from PIL import Image
import time

def DoF(frames, fps, downsample_rate=1, align_timebase=False):
    """Difference of Frames (DoF) respiratory signal extraction.

    Parameters
    ----------
    frames : sequence of ROI frames (n frames)
    fps    : frames per second
    align_timebase : if True, prepend a zero sample so output has len(frames)
                     samples, aligned with per-frame roi_stats_t.
                     Default False preserves original n-1 transition samples.

    Returns
    -------
    sig     : respiratory signal (n-1 or n samples depending on align_timebase)
    elapsed : computation time in seconds

    Note on timebase: By default returns n-1 transition samples (sig[t]
    represents motion between frame t and frame t+1). When used with the
    QROBF filter, roi_stats_t[t] (frame-based) is 1 frame ahead of sig[t].
    Set align_timebase=True to prepend a zero and remove this lag.
    """
    print("\nEstimating Respiration Waveform via Difference of Frames (DoF)...\n")
    start = time.time()
    # Convert PIL Images to numpy and resize to consistent shape
    converted = []
    ref_shape = None
    for f in frames:
        f_np = np.array(f) if not isinstance(f, np.ndarray) else f
        if ref_shape is None:
            ref_shape = f_np.shape
        elif f_np.shape != ref_shape:
            f_np = cv.resize(f_np, (ref_shape[1], ref_shape[0]))
        converted.append(f_np)
    frames_np = np.array(converted).reshape(len(converted),-1)
    dof = np.diff(frames_np, axis=0)
    doft = (dof>100).astype(int)
    sig = np.sum(doft, axis=1)
    end = time.time()
    elapsed = end - start
    if align_timebase:
        sig = np.concatenate([[0.0], sig.astype(float)])
    return sig, elapsed

def OF(frames, fps, align_timebase=False):
    """Optical Flow (Farneback) respiratory signal extraction.

    Parameters
    ----------
    frames : sequence of ROI frames (n frames)
    fps    : frames per second
    align_timebase : if True, prepend a zero sample so output has len(frames)
                     samples, aligned with per-frame roi_stats_t.
                     Default False preserves original n-1 transition samples.

    Returns
    -------
    sig     : respiratory signal (n-1 or n samples depending on align_timebase)
    elapsed : computation time in seconds
    """
    print("\nEstimating Respiration Waveform via Optical Flow (OF)...\n")
    median = []
    ref_shape = None  # For resizing inconsistent ROIs
    t = tqdm(frames)
    for i, curr in enumerate(t):
        # Convert PIL Image to numpy if needed
        if not isinstance(curr, np.ndarray):
            curr = np.array(curr)
        # Resize to reference shape if sizes differ between frames
        if ref_shape is None:
            ref_shape = curr.shape[:2]
        elif curr.shape[:2] != ref_shape:
            curr = cv.resize(curr, (ref_shape[1], ref_shape[0]))
        if i == 0:
            prev = curr
            continue
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vert = flow[...,1].flatten()
        median.append(np.median(vert))
        prev = curr
    sig = np.array(median)
    elapsed = t.format_dict["elapsed"]
    if align_timebase:
        sig = np.concatenate([[0.0], sig.astype(float)])
    return sig, elapsed


def profile1D(frames, fps, interp_type, align_timebase=False):
    import scipy
    """1D Profile Cross-Correlation respiratory signal extraction.

    Parameters
    ----------
    frames      : sequence of ROI frames (n frames)
    fps         : frames per second
    interp_type : 'linear', 'quadratic', or 'cubic'
    align_timebase : if True, prepend a zero sample so output has len(frames)
                     samples, aligned with per-frame roi_stats_t.
                     Default False preserves original n-1 transition samples.

    Returns
    -------
    sig     : respiratory signal (n-1 or n samples depending on align_timebase)
    elapsed : computation time in seconds
    """
    assert interp_type == 'linear' or interp_type == 'quadratic' or interp_type == 'cubic', "'interp_type' should be 'linear', 'quadratic' or 'cubic'"
    print("\nEstimating Respiration Waveform via Cross-Correlation of 1D profiles...\n")
    print("Interpolation type is: " + interp_type + '\n')
    dcp = []    #derivatives of chest position
    ref_shape = None

    t = tqdm(frames)
    for i, curr in enumerate(t):
        # Convert PIL Image to numpy if needed
        if not isinstance(curr, np.ndarray):
            curr = np.array(curr)
        # Resize to reference shape if sizes differ between frames
        if ref_shape is None:
            ref_shape = curr.shape[:2]
        elif curr.shape[:2] != ref_shape:
            curr = cv.resize(curr, (ref_shape[1], ref_shape[0]))
        currp = np.diff(0.5*(np.mean(curr, axis=1) + np.std(curr, axis=1)))
        if currp.size == 0:
            continue
        if i == 0:
            prevp = currp
            continue

        if prevp.size == 0:
            prevp = currp
            continue
        xcorr = np.correlate(currp, prevp, mode='full')
        if xcorr.shape[0] < 4:
            prevp = currp
            continue
        safe_interp = interp_type if xcorr.shape[0] > 3 else 'linear'
        f = scipy.interpolate.interp1d(np.arange(xcorr.shape[0]), xcorr, safe_interp)
        xvals = np.linspace(0, xcorr.shape[0]-1, xcorr.shape[0]*100)
        xcorr_interp = f(xvals)

        disp = np.argmax(xcorr_interp) / (1./fps)
        dcp.append(disp)
        prevp = currp

    sig = np.array(dcp)
    elapsed = t.format_dict["elapsed"]
    if align_timebase:
        sig = np.concatenate([[0.0], sig.astype(float)])
    return sig, elapsed
