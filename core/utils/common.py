from __future__ import division
import numpy as np
try:
    import mediapipe as mp
except Exception:  # pragma: no cover - runtime dependency fallback
    mp = None
from scipy import signal
from matplotlib import pyplot as plt
from scipy.signal._arraytools import even_ext
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, absolute, linspace, log10, logical_and, average, diff, correlate
from scipy.signal.windows import blackmanharris
from scipy.signal import fftconvolve
import sys
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable or []
            self.format_dict = {'elapsed': 0.0}
        def __iter__(self):
            return iter(self.iterable)
        @staticmethod
        def write(msg):
            print(msg)
from PIL import Image
import cv2 as cv
import re 


_MP_SOLUTIONS_WARNED = False


def _get_mp_solutions():
    """Return mediapipe.solutions module when available, else None."""
    global _MP_SOLUTIONS_WARNED
    if mp is None:
        if not _MP_SOLUTIONS_WARNED:
            print("[WARN] mediapipe is not available. Using heuristic ROI fallback.")
            _MP_SOLUTIONS_WARNED = True
        return None
    solutions = getattr(mp, "solutions", None)
    if solutions is None and not _MP_SOLUTIONS_WARNED:
        print("[WARN] mediapipe.solutions is unavailable in this build. Using heuristic ROI fallback.")
        _MP_SOLUTIONS_WARNED = True
    return solutions


def _clip_bbox(xmin, xmax, ymin, ymax, width, height):
    xmin = max(int(round(xmin)), 0)
    xmax = min(int(round(xmax)), width)
    ymin = max(int(round(ymin)), 0)
    ymax = min(int(round(ymax)), height)
    if xmax <= xmin:
        xmax = min(width, xmin + 1)
    if ymax <= ymin:
        ymax = min(height, ymin + 1)
    return [xmin, xmax, ymin, ymax]


def _fallback_face_bbox(img):
    """Top-center face-like fallback when detection is unavailable."""
    image_height, image_width = img.shape[:2]
    side = int(0.35 * min(image_width, image_height))
    cx = image_width * 0.5
    cy = image_height * 0.28
    return _clip_bbox(cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2, image_width, image_height)


def _fallback_chest_bbox(img, face_bbox=None):
    """Deterministic chest ROI fallback derived from frame geometry (and face when available)."""
    image_height, image_width = img.shape[:2]
    if face_bbox is not None:
        fxmin, fxmax, fymin, fymax = face_bbox
        f_w = max(1.0, float(fxmax - fxmin))
        f_h = max(1.0, float(fymax - fymin))
        cx = (fxmin + fxmax) / 2.0
        cy = fymax + 0.9 * f_h
        chest_w = min(image_width * 0.85, 1.9 * f_w)
        chest_h = max(6.0, 0.23 * chest_w)
    else:
        cx = image_width * 0.5
        cy = image_height * 0.60
        chest_w = image_width * 0.40
        chest_h = max(6.0, image_height * 0.09)
    return _clip_bbox(cx - chest_w / 2, cx + chest_w / 2, cy - chest_h / 2, cy + chest_h / 2, image_width, image_height)


def plot_time_and_freq(list_of_sigs):
    plt.figure()
    n_sigs = len(list_of_sigs)
    plt.subplot(2,n_sigs//2, 1)
    plt.plot(list_of_sigs[0], c='r')
    plt.grid()
    for i in range(2,n_sigs+1):
        plt.subplot(2,n_sigs//2, i)
        plt.plot(list_of_sigs[i-1], c='b')
        plt.grid()
    #WELCH
    #GT Welch plot
    fps = 1000
    win_size = 30 
    nyquistF = fps/2
    fRes = 0.1
    nFFT = max(2048, (60*2*nyquistF) / fRes)
    minF = 0.1
    maxF = 0.5
    plt.figure()
    plt.subplot(2,n_sigs//2,1)
    F, P = signal.welch(list_of_sigs[0], nperseg=win_size*fps, noverlap=fps*(win_size-1), fs=fps, nfft=nFFT)
    plt.plot(F,P)
    plt.axvline(x=0.1, ymin=0, ymax=1, c='r')
    plt.axvline(x=maxF, ymin=0, ymax=1, c='r')
    plt.xlim([0,maxF+0.5])
    plt.title("Max frequency GT: "+str(round(F[np.argmax(P)],2))+" Hz, "+str(round(F[np.argmax(P)]*60,2))+" resp/min")
    for i in range(2,n_sigs+1):
        fps = 25
        win_size = 30 
        nyquistF = fps/2
        fRes = 0.1
        nFFT = max(2048, (60*2*nyquistF) / fRes)
        plt.subplot(2,n_sigs//2,i)
        F, P = signal.welch(list_of_sigs[i-1], nperseg=win_size*fps, noverlap=fps*(win_size-1), fs=fps, nfft=nFFT)
        band = np.argwhere((F > minF) & (F < maxF)).flatten()
        plt.plot(F,P)
        plt.axvline(x=0.1, ymin=0, ymax=1, c='r')
        plt.axvline(x=maxF, ymin=0, ymax=1, c='r')
        plt.xlim([0,maxF+0.5])
        plt.title("Max frequency: "+str(round(F[band][np.argmax(P[band])],2))+" Hz, "+str(round(F[np.argmax(P[band])]*60,2))+" resp/min")
    plt.show()

def get_vid_stats(videoFileName):
    cap = cv.VideoCapture(videoFileName)
    fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return duration, int(fps)

def sort_nicely(l): 
  """ Sort the given list in the way that humans expect. 
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  l.sort( key=alphanum_key ) 
  return l

def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()

def detect_face(img):
    image_height, image_width, _ = img.shape
    solutions = _get_mp_solutions()
    if solutions is None or not hasattr(solutions, "face_detection"):
        return _fallback_face_bbox(img)

    mp_face_detection = solutions.face_detection
    try:
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as face_detection:
            detection_result = face_detection.process(img)

        if detection_result is None or not detection_result.detections:
            return _fallback_face_bbox(img)

        bbox = detection_result.detections[0].location_data.relative_bounding_box
        bbox_pxl = [
            bbox.xmin * image_width,
            bbox.ymin * image_height,
            bbox.width * image_width,
            bbox.height * image_height,
        ]
        xmin = bbox_pxl[0]
        xmax = xmin + bbox_pxl[2]
        ymin = bbox_pxl[1]
        ymax = ymin + bbox_pxl[3]
        centerx = xmax - (xmax - xmin) / 2
        centery = ymax - (ymax - ymin) / 2
        xdist = max(image_width - centerx, centerx)
        ydist = max(image_height - centery, centery)
        d = min(xdist, ydist)
        xmin = centerx - d
        xmax = centerx + d
        ymin = centery - d
        ymax = centery + d
        return _clip_bbox(xmin, xmax, ymin, ymax, image_width, image_height)
    except Exception:
        return _fallback_face_bbox(img)

def get_face_ROI(video_path, **kwargs):
    import cv2
    print("\nExtracting face ROIs...")
    i = 0
    frames = []
    t = tqdm(extract_frames_yield(video_path))
    for frame in t:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (i == 0):
            bbox = detect_face(frame)

        crp = frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
        width = crp.shape[1]
        height = crp.shape[0]
        if width >= height:
            crp = crp[:, max(0,int(width/2)-int(height/2 + 1)):int(height/2)+int(width/2), :]
        else:
            crp = crp[int((height-width)):,:,:]
        frames.append(crp)
        i += 1
    return frames

def get_chest_ROI(video_path, dataset, mp_complexity=2, skip_rate=1):
    print("\nExtracting ROIs...")

    _, fps = get_vid_stats(video_path)
    update_every = max(1, int(skip_rate) if skip_rate is not None else 1)
    i = 0
    solutions = _get_mp_solutions()
    mp_pose = solutions.pose if (solutions is not None and hasattr(solutions, "pose")) else None

    frames = []

    # Run MediaPipe Pose when available; otherwise use deterministic heuristic fallback.
    t = tqdm(extract_frames_yield(video_path))
    if mp_pose is None:
        fallback_bbox = None
        for frame in t:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            if fallback_bbox is None:
                face_bbox = detect_face(frame)
                fallback_bbox = _fallback_chest_bbox(frame, face_bbox=face_bbox)
            im = Image.fromarray(frame)
            left, right, upper, lower = fallback_bbox[0], fallback_bbox[1], fallback_bbox[2], fallback_bbox[3]
            if upper >= lower:
                lower = upper + 1
            if left >= right:
                right = left + 1
            chest = im.crop(box=(left, upper, right, lower))
            frames.append(chest)
            i += 1
    else:
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=mp_complexity) as pose:
            results = None
            patch_width = None
            patch_height = None
            fallback_bbox = None
            last_bbox = None
            for frame in t:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Keep ROI size fixed for OF stability, but refresh center periodically.
                if i == 0 or (i % update_every == 0):
                    results = pose.process(frame)

                image_height, image_width, _ = frame.shape

                # Get landmark.
                if results is not None and results.pose_landmarks is not None:
                    x_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
                    y_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
                    x_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
                    y_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

                    shoulder_span = max(float(abs(x_left - x_right)), image_width * 0.15)
                    if patch_width is None or patch_height is None:
                        patch_width = shoulder_span
                        patch_height = max(patch_width * 0.2, image_height * 0.05)

                    center_x = 0.5 * (x_left + x_right)
                    center_y = min(y_right, y_left)
                    left = center_x - patch_width / 2
                    right = center_x + patch_width / 2
                    upper = center_y - patch_height / 2
                    lower = center_y + patch_height / 2
                    left, right, upper, lower = _clip_bbox(left, right, upper, lower, image_width, image_height)
                    last_bbox = [left, right, upper, lower]
                else:
                    if last_bbox is not None:
                        left, right, upper, lower = last_bbox
                    else:
                        if fallback_bbox is None:
                            face_bbox = detect_face(frame)
                            fallback_bbox = _fallback_chest_bbox(frame, face_bbox=face_bbox)
                        left, right, upper, lower = fallback_bbox[0], fallback_bbox[1], fallback_bbox[2], fallback_bbox[3]
                        last_bbox = [left, right, upper, lower]

                im = Image.fromarray(frame)
                if upper >= lower:
                    lower = upper + 1
                if left >= right:
                    right = left + 1
                chest = im.crop(box=(left, upper, right, lower))
                frames.append(chest)
                i += 1
    elapsed = t.format_dict["elapsed"]
    return frames, fps, elapsed

def Welch_rpm(resp, fps, winsize, minHz=0.1, maxHz=0.4, fRes=0.1):
    """
    This method computes the spectrum of a respiratory signal

    Parameters
    ----------
        resp: the respiratory signal
        fps: the fps of the video from which signal is estimated
        winsize: the window size used to compute spectrum
        minHz: the lower bound for accepted frequencies
        maxHz: the upper bound for accepted frequencies

    Returns
    -------
        the array of frequencies and the corrisponding PSD
    """
    step = 1
    nperseg=fps*winsize
    noverlap=fps*(winsize-step)

    nyquistF = fps/2
    nfft = max(2048, (60*2*nyquistF) / fRes)

    # -- periodogram by Welch
    F, P = signal.welch(resp, nperseg=nperseg, noverlap=noverlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()

    Pfreqs = 60*F[band]
    Power = P[:, band]

    return Pfreqs, Power

def sig_to_RPM(sig, fps, winsize, minHz=0.1, maxHz=0.4):
    sig = [s for s in sig if np.asarray(s).size > 0]
    if len(sig) == 0:
        return np.array([np.nan])
    sig = np.vstack(sig)

    Pfreqs, Power = Welch_rpm(sig, fps, winsize, minHz, maxHz)
    Pmax = np.argmax(Power, axis=1)  # power max
    rpm = Pfreqs[Pmax.squeeze()]

    if (rpm.size == 1):
        return rpm.reshape(1, -1)

    return rpm

def select_component(sig, fps, winsize, minHz=0.1, maxHz=0.4):
    
    cur_pMax = 0

    for d in range(sig.shape[0]):
        Pfreqs, Power = Welch_rpm(sig[d,:][np.newaxis,:], fps, winsize, minHz, maxHz)
        pMax = np.max(Power, axis=1)  # power max
        
        if pMax > cur_pMax:
            cur_pMax = pMax
            cur_d = d

    return sig[cur_d, :][np.newaxis,:]


def average_filter(sig, win_length = 5):
    """
    This method applies to a signal an average filter

    Parameters
    ----------
        sig: the respiratory signal
        win_length: the length of the window used to apply the average filter

    Returns
    -------
        the filtered signal
    """
    res = []
    sig = even_ext(np.array(sig), win_length, axis=-1)
    for i in np.arange(win_length, len(sig)-win_length+1):
        window = np.sum(sig[i-win_length:i+win_length])
        res.append(1/(1+2*win_length)*window)
    return res

def filter_RW(sig, fps, lo=0.08, hi=0.5):
    """
    This method performs posptprocessing steps of fiedler methods; the postprocessing process performs on the signal a normalization, computes the gradient of the signal and applies a band-pass filter

    Parameters
    ----------
        sig: the considered signal
        fps : the fps of the considered video

    Returns
    -------
        the postprocessed signal
    """
    #sig = np.diff(np.asarray(sig), axis=0)
    #sig = np.squeeze(sig)
    if (sig.ndim == 1):
        sig = sig[np.newaxis,:]

    b, a = signal.butter(N=2, Wn=[lo, hi], fs=fps, btype='bandpass')
    padlen = 3 * max(len(a), len(b))
    if sig.shape[-1] <= padlen:
        # Signal too short for filtfilt; return zeros
        return np.zeros_like(sig)
    filtered_sig = signal.filtfilt(b, a, sig)

    return filtered_sig

def butter_lowpass_filter(data, cutoff, fs, order=6):
    """
    This method applies to a signal a butter lowpass filter

    Parameters
    ----------
        data: the respiratory signal
        cutoff: the cutoff frequency
        fs: the sampling frequency
        order: the order of the filter

    Returns
    -------
        the filtered signal
    """
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y

def plot_mask(mask):
    """
    This method plots the mask given as input

    Parameters
    ----------
        mask: the input mask

    Returns
    -------
        the plotted mask
    """
    plt.imshow(mask, interpolation='nearest')
    plt.show()

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_autocorr(sig, fs):
    """Estimate frequency using autocorrelation

    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental

    Cons: Not as accurate, currently has trouble with finding the true peak

    """
    # Calculate autocorrelation and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[int(len(corr)/2):]

    # Find the first low point
    d = diff(corr)
    start, = np.nonzero(np.ravel(d > 0))
    start = start[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable, due to peaks that occur between samples.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def freq_from_crossings(sig, fs):
    """Estimatcorr[len(corr)/2:]e frequency by counting zero crossings

    Pros: Fast, accurate (increasing with data length).  Works well for long low-noise sines, square, triangle, etc.

    Cons: Doesn't work if there are multiple zero crossings per cycle, low-frequency baseline shift, noise, etc.

    """
    # Find all indices right before a rising-edge zero crossing
    indices, = np.nonzero(np.ravel((sig[1:] >= 0) & (sig[:-1] < 0)))

    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    #crossings = indices

    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]

    # Some other interpolation based on neighboring points might be better. Spline, cubic, whatever

    return fs / average(diff(crossings))

def freq_from_fft(sig, fs):
    """Estimate frequency from peak of FFT

    Pros: Accurate, usually even more so than zero crossing counter
    (1000.000003 Hz for 1000 Hz, for instance).  Due to parabolic interpolation
    being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with data length

    Cons: Doesn't find the right value if harmonics are stronger than
    fundamental, which is common.  Better method would try to identify the fundamental

    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(abs(f), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def snr(sig, fs, nperseg, noverlap):
    """
    This method computes the SNR of a signal

    Parameters
    ----------
        sig: the respiratory signal
        fs: the sampling frequency
        nperseg: the length of each segment
        noverlap: the number of points to overlap between segments

    Returns
    -------
        the SNR of the given signal
    """
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    num = 0
    den = 0
    for i in np.arange(len(freqs)):
        if freqs[i]>=0.1 and freqs[i]<=0.4:
            num+=psd[i]
        if freqs[i]>=0 and freqs[i]<=4:
            den+=psd[i]
    if den!=0:
        return num/den
    else:

        return -1

def pad_rgb_signal(sig, fps, win_size):
    """
    This method applies padding to a windowed rgb signal

    Parameters
    ----------
        sig: the respiratory signal
        fps: the sampling frequency
        win_size: the length of each segment

    Returns
    -------
        The padded RGB respiratory signal
    """
    sig = np.swapaxes(sig,0,1)

    nperseg = fps * win_size

    new_sig = []
    for roi in sig:
        red = [frame[0] for frame in roi]
        green = [frame[1] for frame in roi]
        blue = [frame[2] for frame in roi]

        red = even_ext(np.asarray(red), int(nperseg//2), axis=-1)
        green = even_ext(np.asarray(green), int(nperseg//2), axis=-1)
        blue = even_ext(np.asarray(blue), int(nperseg//2), axis=-1)

        new_roi = []
        for i in np.arange(len(red)):
            new_roi.append([red[i], green[i], blue[i]])

        new_sig.append(new_roi)


    return np.swapaxes(new_sig,0,1)

def get_channel(sig, channel):
    """
    This method select from a windowed rgb signal a single channel

    Parameters
    ----------
        sig: the respiratory signal
        channel: the channel index (0:red, 1:green, 2:blue)

    Returns
    -------
        The signal resukting from the selection
    """
    res = []
    for win in sig:
        row = []
        for roi in win:
            row.append(roi[channel])
        res.append(row)
    return res

def get_SNR(RW, reference_rr, fps):
    '''Computes the signal-to-noise ratio of the BVP
    signals according to the method by -- de Haan G. et al., IEEE Transactions on Biomedical Engineering (2013).
    SNR calculated as the ratio (in dB) of power contained within +/- 0.1 Hz
    of the reference heart rate frequency and +/- 0.2 of its first
    harmonic and sum of all other power between 0.5 and 4 Hz.
    Adapted from https://github.com/danmcduff/iphys-toolbox/blob/master/tools/bvpsnr.m
    '''
   
    interv1 = 0.05*60
    
    #Estimations params
    win_size = 30 
    nyquistF_est = fps/2
    fRes = 0.1
    nFFT_est = max(2048, (60*2*nyquistF_est) / fRes)
    minF = 0.05
    maxF = 1.5
   
    F, P = signal.welch(RW, nperseg=win_size*fps, noverlap=fps*(win_size-1), fs=fps, nfft=nFFT_est)
    band = np.argwhere((F > minF) & (F < maxF)).flatten()
    pfreqs = 60*F[band]
    power = P[band]
    GTMask = np.logical_and(pfreqs>=reference_rr-interv1, pfreqs<=reference_rr+interv1)
    FMask = np.logical_not(GTMask)

    SPower = np.sum(power[GTMask])
    allPower = np.sum(power[FMask])
    snr = 10*np.log10(SPower/allPower)

    return snr

def _plot_PSD_snr(pfreqs, power, reference_rr, interv1):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(pfreqs, np.squeeze(p))
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref))]
    plt.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=2)
    
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref-interv1))]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    x1 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    x2 = pfreqs[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    y1 = 0
    y2 = p[np.argmin(np.abs(pfreqs-curr_ref+interv1))]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    plt.grid()
    plt.show()

def sig_windowing(sig, fps, wsize, stride=1):
    """ Performs signal windowing

    Args:
      sig (list/array): full signal
      fps       (float): frames per seconds      
      wsize     (float): size of the window (in seconds)
      stride    (float): stride (in seconds)

    Returns:
      win_sig (list): windowed signal
      timesES (list): times of (centers) windows 
    """
    sig = np.array(sig).squeeze()
    block_idx, timesES = sliding_straded_win_idx(sig.shape[0], wsize, stride, fps)
    sig_win  = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame+1])
        sig_win.append(wind_signal[np.newaxis, :])

    return sig_win, timesES

def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    This method is used to compute the indices for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one is the center of a window.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)
