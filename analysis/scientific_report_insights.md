# Empirical Analysis of rPPG Signal Properties and Noise Characteristics

**Date:** 2025-12-24
**Dataset:** COHFACE (N=160, Full Dataset)
**Objective:** To provide quantitative evidence that standard linear filtering assumptions are theoretically insufficient for remote physiological sensing due to non-Gaussian noise and non-linear distortions.

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

## 3. Conclusion for Empirical Analysis
The analysis proves that **rPPG noise is inherently non-Gaussian and time-varying**.
*   **Standard Linear Filters** fail because they penalize outliers quadratically (assuming Gaussianity), leading to tracking instability during noise bursts.
*   **Robust Framework Requirements**: Future model designs must incorporate mechanisms to dynamically modulate the "trust" placed in observations based on the instantaneous statistics of the signal residuals and the identified noise regimes (Leptokurtic/Harmonic).
