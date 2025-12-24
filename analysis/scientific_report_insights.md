# Empirical Justification for Attention-Guided Adaptive Kalman Filter (AG-AKF)

**Date:** 2025-12-24
**Dataset:** COHFACE (N=160, Full Dataset)
**Objective:** To provide quantitative evidence that standard linear filtering (KF, UKF) is theoretically insufficient for remote physiological sensing due to Non-Gaussian noise and Non-linear distortions.

---

## 1. Quantitative Evidence (The "Smoking Gun")

We analyzed 160 video samples using 5 different motion extraction methods. The results challenge the fundamental assumption of Gaussian noise ($L_2$ norm optimality) in standard Kalman Filters.

| Method | Raw Kurtosis | THD (Non-linearity) | Impulse Rate (Outliers) | **Noise Kurtosis** |
| :--- | :--- | :--- | :--- | :--- |
| **OF_Farneback** | 0.38 | 0.3107 | 1.31% | **6.53** |
| **DoF** | 0.38 | **0.9468** | 2.22% | **9.13** |
| **Profile1D_Linear** | 0.38 | 0.4037 | 1.37% | **11.53** |
| **Profile1D_Quad** | 0.38 | 0.2110 | 0.71% | **2.01** |
| **Profile1D_Cubic** | 0.38 | 0.2173 | 0.73% | **2.10** |

> **Note:** A Gaussian distribution has a Kurtosis of 0. Values > 1 indicate heavy-tailed (Leptokurtic) distributions prone to outliers.

---

## 2. Phenomenological Interpretation

### A. The Failure of Linear Interpolation (Kurtosis = 11.53)
*   **Observation**: `Profile1D_Linear` exhibits extreme leptokurtic noise behavior.
*   **Cause**: Linear interpolation in 1D profiles introduces "aliasing steps" when the sub-pixel motion is smaller than the spatial grid. This results in impulsive "jumps" in the observation signal that are not biological.
*   **Scientific Insight**: Lightweight models (often required for mobile rPPG) usually rely on cheaper interpolation (Linear). **Therefore, a robust filter is mandatory** to handle the resulting non-Gaussian noise.

### B. Harmonic Non-Linearity in Physiological Signals (THD ~ 0.3-0.4)
*   **Observation**: Even robust methods like `OF_Farneback` and `Profile1D_Linear` exhibit significant Total Harmonic Distortion (THD $\approx$ 0.31 - 0.40).
*   **Cause**: Respiratory motion is not a perfect sinusoid. Ideally, it is separate from gross body motion, but in rPPG, the chest wall expansion projects non-linearly onto the 2D image plane. This creates structural 2nd and 3rd harmonics in the measured signal.
*   **Scientific Insight**: We must treat these harmonics as "Signal," not just noise. The filter must track the fundamental frequency while actively disentangling it from these induced harmonics to prevent frequency doubling errors.
*   **Excluded Metric (DoF)**: While `DoF` (Difference of Frames) showed extreme non-linearity (THD 0.95), it is excluded from this modeling. Our prior research demonstrated that DoF captures pixel intensity flux rather than the geometric displacement required for respiratory mechanics. Thus, its high distortion is an artifact of the method, not the physiology.

---

## 3. Translation to State-Space Model (SSM) Design

To overcome the identified limitations, the filter design must not be static. Instead, it should explicitly utilize the analysis metadata to dynamically configure the **State-Space Equations**.

### Meta-Adaptive Mapping Strategy

We propose mapping the four key analysis metrics directly to the Kalman Filter's structural parameters:

| Metadata Source | Analyzed Value (Example) | Target SSM Parameter | Adaptation Logic (Mechanism) |
| :--- | :--- | :--- | :--- |
| **FPS (Temporal)** | N/A (Variable) | **State Transition ($F_t$)** | **Exact Time-Step Scaling**: <br> $\Delta t = 1/\text{FPS}$ is injected into the rotation matrix. This ensures frequency states ($f$) remain physically accurate (Hz) regardless of camera speed. |
| **Raw Kurtosis** | High (> 3.0), e.g., 11.53 | **Obs. Noise Covariance ($R_t$)** | **Regime Switching**: <br> - If Kurtosis $< 3$: Use standard static $R$. <br> - If Kurtosis $> 3$: Activate **Robust Mode** (e.g., Huber/Tukey loss) to exponentially penalize large residuals. |
| **THD (Spectrum)** | High (> 0.5), e.g., 0.95 | **Observation Matrix ($H$)** | **Harmonic Expansion**: <br> - Instead of $H=[1, 0]$, expand to $H=[1, 0, 1, 0, \dots]$ to explicitly model and subtract harmonics ($2f, 3f$), preventing frequency locking to artifacts. |
| **Impulse Rate** | High (> 1.0%) | **Process Noise Covariance ($Q_t$)** | **Inertia Modulation**: <br> - In high-impulse regimes, reduce $Q$ for the frequency state to increase "momentum" and resist sudden, physically impossible jumps in breathing rate. |

### Implication for Filter Architecture
The resulting model is not just a "Kalman Filter with Attention," but a **Meta-Analytic Filter Framework**.
1.  **Initialization Phase**: Read `obs_meta.json` (Kurtosis, THD) to select the optimal model structure ($H$) and loss function type.
2.  **Update Phase**: Use real-time residuals ($y_t - \hat{y}_t$) to dynamically adjust trust levels ($R_t$) based on the pre-determined noise regime.

---

## 4. Conclusion for Scientific Reports
The analysis proves that **rPPG noise is inherently Non-Gaussian and Time-Varying**.
*   **Standard Filters (KF-std)** fail because they penalize outliers quadratically (assuming Gaussianity), leading to tracking instability during noise bursts.
*   **AG-AKF** is theoretically justified as it dynamically modulates the "trust" ($R_t^{-1}$) placed in observations based on the instantaneous statistics of the signal residuals.
