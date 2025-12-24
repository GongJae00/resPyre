import os
import numpy as np
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    go = None

def _save_or_show(fig, save_path=None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
    else:
        fig.show()

def plot_breathing_signals(times, signal_est, signal_gt=None, title="Breathing Signal", save_path=None):
    if go is None:
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=signal_est, mode='lines', name='Estimated'))
    
    if signal_gt is not None:
        # Normalize for visual comparison if needed, or assume pre-normalized
        fig.add_trace(go.Scatter(x=times, y=signal_gt, mode='lines', name='Ground Truth', line=dict(dash='dot')))
    
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude")
    _save_or_show(fig, save_path)

def plot_bpm_tracking(times, bpm_est, bpm_gt=None, title="BPM Tracking", save_path=None):
    if go is None:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=bpm_est, mode='lines', name='Estimated BPM'))
    
    if bpm_gt is not None:
         fig.add_trace(go.Scatter(x=times, y=bpm_gt, mode='lines', name='GT BPM', line=dict(color='gray', dash='dash')))

    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="BPM")
    _save_or_show(fig, save_path)

def plot_comprehensive_result(times, signal_est, bpm_est, signal_gt=None, bpm_gt=None, title="Analysis Result", save_path=None):
    if go is None:
        return
        
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Breathing Signal", "Respiratory Rate (BPM)"))
    
    # Signal
    fig.add_trace(go.Scatter(x=times, y=signal_est, name='Est Signal', line=dict(color='blue')), row=1, col=1)
    if signal_gt is not None:
        fig.add_trace(go.Scatter(x=times, y=signal_gt, name='GT Signal', line=dict(color='orange', dash='dot')), row=1, col=1)
        
    # BPM
    fig.add_trace(go.Scatter(x=times, y=bpm_est, name='Est BPM', line=dict(color='green')), row=2, col=1)
    if bpm_gt is not None:
         fig.add_trace(go.Scatter(x=times, y=bpm_gt, name='GT BPM', line=dict(color='red', dash='dot')), row=2, col=1)
         
    fig.update_layout(title=title, height=800)
    _save_or_show(fig, save_path)
