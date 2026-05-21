# PARH Rate-Source Decomposition

- data: `results/final_full_validation/mahnob_tailaligned/data`
- purpose: separate native PARH oscillator rate from external target-computable readout and final reported rate.
- interpretation: if `final_track_hz` is much better than `native_smoothed_track_hz`, the current performance is readout-carried rather than state-carried.

## Source Summary

| method | rate_source | n_trials | MAE_median | RMSE_median | PearsonR_median | track_hz_median | external_output_blend_mean | external_posterior_blend_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parh_ossm | external_output_rate_t | 513 | 2.390 | 2.960 | 0.230 | 0.260 | 0.229 | 0.047 |
| parh_ossm | final_track_hz | 525 | 2.410 | 2.920 | 0.230 | 0.256 | 0.228 | 0.047 |
| parh_ossm | external_rate_posterior_mean_t | 525 | 2.460 | 2.920 | 0.180 | 0.250 | 0.228 | 0.047 |
| parh_ossm | state_freq_t | 525 | 2.520 | 3.010 | 0.230 | 0.261 | 0.228 | 0.047 |
| parh_ossm | external_rate_posterior_mode_t | 525 | 2.570 | 3.070 | 0.190 | 0.272 | 0.228 | 0.047 |
| parh_ossm | native_causal_track_hz | 525 | 2.590 | 3.060 | 0.150 | 0.253 | 0.228 | 0.047 |
| parh_ossm | native_smoothed_track_hz | 525 | 2.590 | 3.100 | 0.200 | 0.258 | 0.228 | 0.047 |

## Per-Trial Rows

| video | data_file | method | rate_source | MAE | RMSE | MAPE | PearsonR | n_windows | est_bpm_avg | gt_bpm_avg | track_hz_median | track_hz_std | external_output_blend_mean | external_posterior_blend_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mahnob_10 | mahnob_10.pkl | parh_ossm | external_output_rate_t | 2.580 | 2.980 | 19.510 | 0.570 | 107 | 16.239 | 13.681 | 0.279 | 0.034 | 0.199 | 0.068 |
| mahnob_10 | mahnob_10.pkl | parh_ossm | external_rate_posterior_mean_t | 2.030 | 2.330 | 15.810 | 0.510 | 107 | 15.619 | 13.681 | 0.256 | 0.020 | 0.199 | 0.068 |
| mahnob_10 | mahnob_10.pkl | parh_ossm | external_rate_posterior_mode_t | 3.670 | 3.870 | 28.070 | 0.590 | 107 | 17.356 | 13.681 | 0.286 | 0.021 | 0.199 | 0.068 |
| mahnob_10 | mahnob_10.pkl | parh_ossm | final_track_hz | 2.470 | 2.830 | 18.880 | 0.510 | 107 | 16.097 | 13.681 | 0.268 | 0.037 | 0.199 | 0.068 |
| mahnob_10 | mahnob_10.pkl | parh_ossm | native_causal_track_hz | 2.420 | 2.820 | 18.690 | 0.330 | 107 | 15.932 | 13.681 | 0.270 | 0.087 | 0.199 | 0.068 |
| mahnob_10 | mahnob_10.pkl | parh_ossm | native_smoothed_track_hz | 2.680 | 2.980 | 20.720 | 0.460 | 107 | 16.327 | 13.681 | 0.271 | 0.054 | 0.199 | 0.068 |
| mahnob_10 | mahnob_10.pkl | parh_ossm | state_freq_t | 2.740 | 3.130 | 21.160 | 0.300 | 107 | 16.331 | 13.681 | 0.276 | 0.025 | 0.199 | 0.068 |
| mahnob_1042 | mahnob_1042.pkl | parh_ossm | external_output_rate_t | 3.360 | 3.790 | 26.090 | 0.290 | 78 | 14.185 | 14.231 | 0.242 | 0.018 | 0.230 | 0.025 |
| mahnob_1042 | mahnob_1042.pkl | parh_ossm | external_rate_posterior_mean_t | 3.500 | 4.010 | 25.310 | 0.170 | 78 | 13.272 | 14.231 | 0.222 | 0.010 | 0.230 | 0.025 |
| mahnob_1042 | mahnob_1042.pkl | parh_ossm | external_rate_posterior_mode_t | 3.090 | 3.470 | 24.280 | 0.660 | 78 | 14.497 | 14.231 | 0.245 | 0.021 | 0.230 | 0.025 |
| mahnob_1042 | mahnob_1042.pkl | parh_ossm | final_track_hz | 3.280 | 3.660 | 25.850 | 0.430 | 78 | 14.334 | 14.231 | 0.241 | 0.027 | 0.230 | 0.025 |
| mahnob_1042 | mahnob_1042.pkl | parh_ossm | native_causal_track_hz | 3.020 | 3.530 | 23.100 | 0.510 | 78 | 13.720 | 14.231 | 0.230 | 0.095 | 0.230 | 0.025 |
| mahnob_1042 | mahnob_1042.pkl | parh_ossm | native_smoothed_track_hz | 3.030 | 3.410 | 25.190 | 0.570 | 78 | 14.899 | 14.231 | 0.238 | 0.059 | 0.230 | 0.025 |
| mahnob_1042 | mahnob_1042.pkl | parh_ossm | state_freq_t | 2.940 | 3.340 | 22.690 | 0.960 | 78 | 14.159 | 14.231 | 0.234 | 0.016 | 0.230 | 0.025 |
| mahnob_1044 | mahnob_1044.pkl | parh_ossm | external_output_rate_t | 2.250 | 2.850 | 22.670 | -0.050 | 54 | 13.091 | 11.386 | 0.230 | 0.032 | 0.247 | 0.042 |
| mahnob_1044 | mahnob_1044.pkl | parh_ossm | external_rate_posterior_mean_t | 1.780 | 2.240 | 17.810 | 0.380 | 54 | 12.729 | 11.386 | 0.217 | 0.024 | 0.247 | 0.042 |
| mahnob_1044 | mahnob_1044.pkl | parh_ossm | external_rate_posterior_mode_t | 3.380 | 3.860 | 33.090 | 0.320 | 54 | 14.765 | 11.386 | 0.247 | 0.027 | 0.247 | 0.042 |
| mahnob_1044 | mahnob_1044.pkl | parh_ossm | final_track_hz | 1.920 | 2.360 | 18.700 | 0.110 | 54 | 12.308 | 11.386 | 0.218 | 0.036 | 0.247 | 0.042 |
| mahnob_1044 | mahnob_1044.pkl | parh_ossm | native_causal_track_hz | 1.580 | 1.960 | 15.290 | 0.290 | 54 | 11.800 | 11.386 | 0.196 | 0.085 | 0.247 | 0.042 |
| mahnob_1044 | mahnob_1044.pkl | parh_ossm | native_smoothed_track_hz | 1.630 | 1.990 | 15.360 | 0.450 | 54 | 12.052 | 11.386 | 0.210 | 0.057 | 0.247 | 0.042 |
| mahnob_1044 | mahnob_1044.pkl | parh_ossm | state_freq_t | 1.420 | 1.680 | 13.210 | 0.550 | 54 | 11.616 | 11.386 | 0.195 | 0.038 | 0.247 | 0.042 |
| mahnob_1046 | mahnob_1046.pkl | parh_ossm | external_output_rate_t | 3.630 | 4.150 | 18.250 | 0.940 | 67 | 14.961 | 18.476 | 0.257 | 0.037 | 0.242 | 0.041 |
| mahnob_1046 | mahnob_1046.pkl | parh_ossm | external_rate_posterior_mean_t | 3.960 | 4.540 | 19.980 | 0.950 | 67 | 14.853 | 18.476 | 0.257 | 0.021 | 0.242 | 0.041 |
| mahnob_1046 | mahnob_1046.pkl | parh_ossm | external_rate_posterior_mode_t | 3.510 | 4.170 | 17.320 | 0.910 | 67 | 14.985 | 18.476 | 0.263 | 0.045 | 0.242 | 0.041 |
| mahnob_1046 | mahnob_1046.pkl | parh_ossm | final_track_hz | 3.690 | 4.390 | 18.310 | 0.880 | 67 | 14.972 | 18.476 | 0.245 | 0.039 | 0.242 | 0.041 |
| mahnob_1046 | mahnob_1046.pkl | parh_ossm | native_causal_track_hz | 3.900 | 4.740 | 19.330 | 0.650 | 67 | 14.966 | 18.476 | 0.242 | 0.088 | 0.242 | 0.041 |
| mahnob_1046 | mahnob_1046.pkl | parh_ossm | native_smoothed_track_hz | 3.810 | 4.650 | 18.780 | 0.730 | 67 | 14.898 | 18.476 | 0.243 | 0.056 | 0.242 | 0.041 |
| mahnob_1046 | mahnob_1046.pkl | parh_ossm | state_freq_t | 3.540 | 4.250 | 18.010 | 0.860 | 67 | 15.649 | 18.476 | 0.257 | 0.018 | 0.242 | 0.041 |
| mahnob_1048 | mahnob_1048.pkl | parh_ossm | external_output_rate_t | 4.740 | 5.350 | 29.820 | -0.880 | 41 | 12.801 | 15.927 | 0.220 | 0.022 | 0.247 | 0.047 |
| mahnob_1048 | mahnob_1048.pkl | parh_ossm | external_rate_posterior_mean_t | 4.310 | 4.880 | 27.410 | -0.830 | 41 | 13.287 | 15.927 | 0.229 | 0.014 | 0.247 | 0.047 |
| mahnob_1048 | mahnob_1048.pkl | parh_ossm | external_rate_posterior_mode_t | 4.340 | 4.930 | 29.090 | -0.990 | 41 | 13.979 | 15.927 | 0.252 | 0.023 | 0.247 | 0.047 |
| mahnob_1048 | mahnob_1048.pkl | parh_ossm | final_track_hz | 4.410 | 5.010 | 27.260 | 0.080 | 41 | 12.739 | 15.927 | 0.225 | 0.033 | 0.247 | 0.047 |
| mahnob_1048 | mahnob_1048.pkl | parh_ossm | native_causal_track_hz | 5.230 | 6.010 | 31.010 | 0.130 | 41 | 11.365 | 15.927 | 0.207 | 0.094 | 0.247 | 0.047 |
| mahnob_1048 | mahnob_1048.pkl | parh_ossm | native_smoothed_track_hz | 4.080 | 4.640 | 25.400 | 0.340 | 41 | 13.040 | 15.927 | 0.224 | 0.050 | 0.247 | 0.047 |
| mahnob_1048 | mahnob_1048.pkl | parh_ossm | state_freq_t | 4.180 | 4.860 | 25.300 | 0.370 | 41 | 12.659 | 15.927 | 0.218 | 0.035 | 0.247 | 0.047 |
| mahnob_1050 | mahnob_1050.pkl | parh_ossm | external_output_rate_t | 3.220 | 3.500 | 24.570 | 0.130 | 87 | 15.954 | 14.378 | 0.264 | 0.010 | 0.178 | 0.053 |
| mahnob_1050 | mahnob_1050.pkl | parh_ossm | external_rate_posterior_mean_t | 3.010 | 3.250 | 22.500 | 0.270 | 87 | 15.473 | 14.378 | 0.259 | 0.010 | 0.178 | 0.053 |
| mahnob_1050 | mahnob_1050.pkl | parh_ossm | external_rate_posterior_mode_t | 3.390 | 3.720 | 26.190 | -0.010 | 87 | 16.286 | 14.378 | 0.270 | 0.015 | 0.178 | 0.053 |
| mahnob_1050 | mahnob_1050.pkl | parh_ossm | final_track_hz | 3.320 | 3.580 | 24.510 | -0.160 | 87 | 15.507 | 14.378 | 0.255 | 0.025 | 0.178 | 0.053 |
| mahnob_1050 | mahnob_1050.pkl | parh_ossm | native_causal_track_hz | 3.420 | 3.730 | 24.900 | -0.420 | 87 | 15.270 | 14.378 | 0.248 | 0.086 | 0.178 | 0.053 |
| mahnob_1050 | mahnob_1050.pkl | parh_ossm | native_smoothed_track_hz | 3.480 | 3.770 | 25.330 | -0.170 | 87 | 15.339 | 14.378 | 0.254 | 0.051 | 0.178 | 0.053 |
| mahnob_1050 | mahnob_1050.pkl | parh_ossm | state_freq_t | 3.110 | 3.350 | 23.270 | 0.140 | 87 | 15.596 | 14.378 | 0.263 | 0.019 | 0.178 | 0.053 |
| mahnob_1052 | mahnob_1052.pkl | parh_ossm | external_output_rate_t | 2.360 | 3.370 | 13.130 | 0.910 | 69 | 12.642 | 14.848 | 0.202 | 0.026 | 0.266 | 0.033 |
| mahnob_1052 | mahnob_1052.pkl | parh_ossm | external_rate_posterior_mean_t | 2.690 | 3.760 | 15.170 | 0.900 | 69 | 12.673 | 14.848 | 0.214 | 0.015 | 0.266 | 0.033 |
| mahnob_1052 | mahnob_1052.pkl | parh_ossm | external_rate_posterior_mode_t | 2.210 | 2.910 | 13.290 | 0.910 | 69 | 13.679 | 14.848 | 0.232 | 0.024 | 0.266 | 0.033 |
| mahnob_1052 | mahnob_1052.pkl | parh_ossm | final_track_hz | 2.650 | 3.700 | 14.860 | 0.930 | 69 | 12.531 | 14.848 | 0.207 | 0.030 | 0.266 | 0.033 |
| mahnob_1052 | mahnob_1052.pkl | parh_ossm | native_causal_track_hz | 3.680 | 4.620 | 21.760 | 0.790 | 69 | 11.341 | 14.848 | 0.179 | 0.093 | 0.266 | 0.033 |
| mahnob_1052 | mahnob_1052.pkl | parh_ossm | native_smoothed_track_hz | 3.090 | 4.000 | 18.290 | 0.790 | 69 | 12.670 | 14.848 | 0.205 | 0.056 | 0.266 | 0.033 |
| mahnob_1052 | mahnob_1052.pkl | parh_ossm | state_freq_t | 2.780 | 3.910 | 15.530 | 0.850 | 69 | 12.267 | 14.848 | 0.192 | 0.022 | 0.266 | 0.033 |
| mahnob_1054 | mahnob_1054.pkl | parh_ossm | external_output_rate_t | 3.860 | 4.230 | 32.340 | -0.830 | 57 | 12.183 | 12.647 | 0.190 | 0.053 | 0.370 | 0.024 |
| mahnob_1054 | mahnob_1054.pkl | parh_ossm | external_rate_posterior_mean_t | 2.750 | 3.680 | 26.890 | -0.250 | 57 | 14.906 | 12.647 | 0.261 | 0.020 | 0.370 | 0.024 |
| mahnob_1054 | mahnob_1054.pkl | parh_ossm | external_rate_posterior_mode_t | 4.520 | 5.190 | 40.100 | -0.920 | 57 | 13.749 | 12.647 | 0.220 | 0.060 | 0.370 | 0.024 |
| mahnob_1054 | mahnob_1054.pkl | parh_ossm | final_track_hz | 2.670 | 3.430 | 25.440 | -0.300 | 57 | 14.121 | 12.647 | 0.242 | 0.037 | 0.370 | 0.024 |
| mahnob_1054 | mahnob_1054.pkl | parh_ossm | native_causal_track_hz | 4.250 | 4.590 | 38.010 | 0.800 | 57 | 16.897 | 12.647 | 0.289 | 0.090 | 0.370 | 0.024 |
| mahnob_1054 | mahnob_1054.pkl | parh_ossm | native_smoothed_track_hz | 3.910 | 4.180 | 34.360 | 0.840 | 57 | 16.554 | 12.647 | 0.291 | 0.058 | 0.370 | 0.024 |
| mahnob_1054 | mahnob_1054.pkl | parh_ossm | state_freq_t | 4.260 | 4.630 | 38.550 | 0.940 | 57 | 16.911 | 12.647 | 0.271 | 0.020 | 0.370 | 0.024 |
| mahnob_1056 | mahnob_1056.pkl | parh_ossm | external_output_rate_t | 3.530 | 4.450 | 34.460 | 0.570 | 82 | 17.034 | 13.600 | 0.284 | 0.014 | 0.254 | 0.037 |
| mahnob_1056 | mahnob_1056.pkl | parh_ossm | external_rate_posterior_mean_t | 2.860 | 3.890 | 29.040 | 0.370 | 82 | 16.081 | 13.600 | 0.264 | 0.010 | 0.254 | 0.037 |
| mahnob_1056 | mahnob_1056.pkl | parh_ossm | external_rate_posterior_mode_t | 3.700 | 4.540 | 35.130 | 0.330 | 82 | 17.016 | 13.600 | 0.273 | 0.029 | 0.254 | 0.037 |
| mahnob_1056 | mahnob_1056.pkl | parh_ossm | final_track_hz | 3.110 | 4.030 | 30.740 | 0.470 | 82 | 16.432 | 13.600 | 0.270 | 0.031 | 0.254 | 0.037 |
| mahnob_1056 | mahnob_1056.pkl | parh_ossm | native_causal_track_hz | 3.000 | 3.700 | 28.570 | 0.030 | 82 | 15.194 | 13.600 | 0.256 | 0.100 | 0.254 | 0.037 |
| mahnob_1056 | mahnob_1056.pkl | parh_ossm | native_smoothed_track_hz | 3.060 | 3.960 | 30.090 | 0.120 | 82 | 15.939 | 13.600 | 0.263 | 0.064 | 0.254 | 0.037 |
| mahnob_1056 | mahnob_1056.pkl | parh_ossm | state_freq_t | 3.070 | 4.190 | 31.210 | 0.050 | 82 | 16.339 | 13.600 | 0.273 | 0.018 | 0.254 | 0.037 |
| mahnob_1058 | mahnob_1058.pkl | parh_ossm | external_output_rate_t | 4.190 | 5.200 | 24.620 | 0.580 | 115 | 11.555 | 15.140 | 0.192 | 0.022 | 0.264 | 0.026 |
| mahnob_1058 | mahnob_1058.pkl | parh_ossm | external_rate_posterior_mean_t | 2.980 | 3.520 | 19.080 | 0.870 | 115 | 13.421 | 15.140 | 0.225 | 0.027 | 0.264 | 0.026 |
| mahnob_1058 | mahnob_1058.pkl | parh_ossm | external_rate_posterior_mode_t | 4.180 | 5.000 | 25.250 | 0.540 | 115 | 11.700 | 15.140 | 0.183 | 0.029 | 0.264 | 0.026 |
| mahnob_1058 | mahnob_1058.pkl | parh_ossm | final_track_hz | 3.810 | 4.640 | 22.660 | 0.860 | 115 | 11.913 | 15.140 | 0.195 | 0.025 | 0.264 | 0.026 |
| mahnob_1058 | mahnob_1058.pkl | parh_ossm | native_causal_track_hz | 3.500 | 4.250 | 20.720 | 0.900 | 115 | 12.076 | 15.140 | 0.188 | 0.086 | 0.264 | 0.026 |
| mahnob_1058 | mahnob_1058.pkl | parh_ossm | native_smoothed_track_hz | 3.150 | 3.830 | 19.000 | 0.850 | 115 | 12.436 | 15.140 | 0.199 | 0.056 | 0.264 | 0.026 |
| mahnob_1058 | mahnob_1058.pkl | parh_ossm | state_freq_t | 3.270 | 3.860 | 19.870 | 0.910 | 115 | 12.450 | 15.140 | 0.191 | 0.033 | 0.264 | 0.026 |
| mahnob_1060 | mahnob_1060.pkl | parh_ossm | external_output_rate_t | 3.810 | 4.230 | 44.510 | 0.640 | 86 | 13.538 | 9.777 | 0.237 | 0.033 | 0.333 | 0.030 |
| mahnob_1060 | mahnob_1060.pkl | parh_ossm | external_rate_posterior_mean_t | 4.480 | 4.770 | 51.510 | 0.800 | 86 | 14.261 | 9.777 | 0.253 | 0.029 | 0.333 | 0.030 |
| mahnob_1060 | mahnob_1060.pkl | parh_ossm | external_rate_posterior_mode_t | 4.520 | 4.860 | 52.860 | 0.890 | 86 | 14.302 | 9.777 | 0.239 | 0.031 | 0.333 | 0.030 |
| mahnob_1060 | mahnob_1060.pkl | parh_ossm | final_track_hz | 4.560 | 4.900 | 52.840 | 0.760 | 86 | 14.340 | 9.777 | 0.247 | 0.025 | 0.333 | 0.030 |
| mahnob_1060 | mahnob_1060.pkl | parh_ossm | native_causal_track_hz | 5.720 | 6.170 | 66.570 | 0.390 | 86 | 15.496 | 9.777 | 0.248 | 0.079 | 0.333 | 0.030 |
| mahnob_1060 | mahnob_1060.pkl | parh_ossm | native_smoothed_track_hz | 5.840 | 6.250 | 67.770 | 0.450 | 86 | 15.620 | 9.777 | 0.256 | 0.044 | 0.333 | 0.030 |
| mahnob_1060 | mahnob_1060.pkl | parh_ossm | state_freq_t | 6.340 | 6.950 | 74.280 | -0.070 | 86 | 16.121 | 9.777 | 0.244 | 0.029 | 0.333 | 0.030 |
| mahnob_1062 | mahnob_1062.pkl | parh_ossm | external_output_rate_t | 3.500 | 4.230 | 28.860 | -0.150 | 90 | 14.632 | 14.675 | 0.255 | 0.029 | 0.210 | 0.039 |
| mahnob_1062 | mahnob_1062.pkl | parh_ossm | external_rate_posterior_mean_t | 3.350 | 3.920 | 25.900 | 0.050 | 90 | 13.909 | 14.675 | 0.231 | 0.023 | 0.210 | 0.039 |
| mahnob_1062 | mahnob_1062.pkl | parh_ossm | external_rate_posterior_mode_t | 3.530 | 4.160 | 28.790 | -0.030 | 90 | 14.756 | 14.675 | 0.254 | 0.036 | 0.210 | 0.039 |
