# PARH Rate-Source Decomposition

- data: `results/final_full_validation/cohface/data`
- purpose: separate native PARH oscillator rate from external target-computable readout and final reported rate.
- interpretation: if `final_track_hz` is much better than `native_smoothed_track_hz`, the current performance is readout-carried rather than state-carried.

## Source Summary

| method | rate_source | n_trials | MAE_median | RMSE_median | PearsonR_median | track_hz_median | external_output_blend_mean | external_posterior_blend_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parh_ossm | native_smoothed_track_hz | 160 | 0.195 | 0.255 | 0.825 | 0.217 | 0.105 | 0.087 |
| parh_ossm | native_causal_track_hz | 160 | 0.310 | 0.380 | 0.705 | 0.217 | 0.105 | 0.087 |
| parh_ossm | final_track_hz | 160 | 0.335 | 0.410 | 0.855 | 0.219 | 0.105 | 0.087 |
| parh_ossm | external_output_rate_t | 160 | 0.960 | 1.055 | 0.755 | 0.229 | 0.105 | 0.087 |
| parh_ossm | state_freq_t | 160 | 1.090 | 1.175 | 0.480 | 0.230 | 0.105 | 0.087 |
| parh_ossm | external_rate_posterior_mode_t | 160 | 1.145 | 1.220 | 0.720 | 0.237 | 0.105 | 0.087 |
| parh_ossm | external_rate_posterior_mean_t | 160 | 1.315 | 1.400 | 0.690 | 0.229 | 0.105 | 0.087 |

## Per-Trial Rows

| video | data_file | method | rate_source | MAE | RMSE | MAPE | PearsonR | n_windows | est_bpm_avg | gt_bpm_avg | track_hz_median | track_hz_std | external_output_blend_mean | external_posterior_blend_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cohface_10_0 | cohface_10_0.pkl | parh_ossm | external_output_rate_t | 1.390 | 1.450 | 11.880 | -0.060 | 32 | 13.216 | 11.821 | 0.219 | 0.007 | 0.122 | 0.176 |
| cohface_10_0 | cohface_10_0.pkl | parh_ossm | external_rate_posterior_mean_t | 1.590 | 1.650 | 13.570 | -0.050 | 32 | 13.415 | 11.821 | 0.222 | 0.009 | 0.122 | 0.176 |
| cohface_10_0 | cohface_10_0.pkl | parh_ossm | external_rate_posterior_mode_t | 1.610 | 1.650 | 13.670 | -0.010 | 32 | 13.428 | 11.821 | 0.224 | 0.008 | 0.122 | 0.176 |
| cohface_10_0 | cohface_10_0.pkl | parh_ossm | final_track_hz | 0.280 | 0.340 | 2.450 | 0.930 | 32 | 12.099 | 11.821 | 0.201 | 0.012 | 0.122 | 0.176 |
| cohface_10_0 | cohface_10_0.pkl | parh_ossm | native_causal_track_hz | 0.200 | 0.240 | 1.730 | 0.690 | 32 | 11.889 | 11.821 | 0.202 | 0.046 | 0.122 | 0.176 |
| cohface_10_0 | cohface_10_0.pkl | parh_ossm | native_smoothed_track_hz | 0.120 | 0.150 | 1.050 | 0.940 | 32 | 11.867 | 11.821 | 0.199 | 0.017 | 0.122 | 0.176 |
| cohface_10_0 | cohface_10_0.pkl | parh_ossm | state_freq_t | 0.740 | 0.800 | 6.330 | nan | 32 | 12.561 | 11.821 | 0.209 | 0.000 | 0.122 | 0.176 |
| cohface_10_1 | cohface_10_1.pkl | parh_ossm | external_output_rate_t | 0.900 | 0.920 | 6.590 | 0.570 | 32 | 14.582 | 13.683 | 0.242 | 0.003 | 0.118 | 0.142 |
| cohface_10_1 | cohface_10_1.pkl | parh_ossm | external_rate_posterior_mean_t | 0.860 | 0.890 | 6.320 | 0.570 | 32 | 14.545 | 13.683 | 0.241 | 0.006 | 0.118 | 0.142 |
| cohface_10_1 | cohface_10_1.pkl | parh_ossm | external_rate_posterior_mode_t | 1.180 | 1.200 | 8.650 | 0.520 | 32 | 14.862 | 13.683 | 0.248 | 0.001 | 0.118 | 0.142 |
| cohface_10_1 | cohface_10_1.pkl | parh_ossm | final_track_hz | 0.150 | 0.190 | 1.090 | 0.750 | 32 | 13.780 | 13.683 | 0.225 | 0.016 | 0.118 | 0.142 |
| cohface_10_1 | cohface_10_1.pkl | parh_ossm | native_causal_track_hz | 0.190 | 0.210 | 1.370 | 0.590 | 32 | 13.728 | 13.683 | 0.229 | 0.051 | 0.118 | 0.142 |
| cohface_10_1 | cohface_10_1.pkl | parh_ossm | native_smoothed_track_hz | 0.140 | 0.170 | 1.020 | 0.810 | 32 | 13.763 | 13.683 | 0.224 | 0.023 | 0.118 | 0.142 |
| cohface_10_1 | cohface_10_1.pkl | parh_ossm | state_freq_t | 0.570 | 0.620 | 4.220 | 0.180 | 32 | 14.256 | 13.683 | 0.238 | 0.007 | 0.118 | 0.142 |
| cohface_10_2 | cohface_10_2.pkl | parh_ossm | external_output_rate_t | 0.980 | 1.040 | 8.950 | -0.420 | 32 | 12.035 | 11.055 | 0.199 | 0.003 | 0.083 | 0.102 |
| cohface_10_2 | cohface_10_2.pkl | parh_ossm | external_rate_posterior_mean_t | 1.220 | 1.320 | 11.110 | -0.460 | 32 | 12.272 | 11.055 | 0.201 | 0.009 | 0.083 | 0.102 |
| cohface_10_2 | cohface_10_2.pkl | parh_ossm | external_rate_posterior_mode_t | 1.060 | 1.130 | 9.700 | -0.400 | 32 | 12.117 | 11.055 | 0.198 | 0.005 | 0.083 | 0.102 |
| cohface_10_2 | cohface_10_2.pkl | parh_ossm | final_track_hz | 0.230 | 0.290 | 2.100 | 0.850 | 32 | 11.268 | 11.055 | 0.189 | 0.012 | 0.083 | 0.102 |
| cohface_10_2 | cohface_10_2.pkl | parh_ossm | native_causal_track_hz | 0.200 | 0.240 | 1.790 | 0.550 | 32 | 11.022 | 11.055 | 0.192 | 0.053 | 0.083 | 0.102 |
| cohface_10_2 | cohface_10_2.pkl | parh_ossm | native_smoothed_track_hz | 0.130 | 0.170 | 1.140 | 0.810 | 32 | 11.071 | 11.055 | 0.189 | 0.020 | 0.083 | 0.102 |
| cohface_10_2 | cohface_10_2.pkl | parh_ossm | state_freq_t | 1.740 | 1.790 | 15.840 | -0.430 | 32 | 12.796 | 11.055 | 0.217 | 0.009 | 0.083 | 0.102 |
| cohface_10_3 | cohface_10_3.pkl | parh_ossm | external_output_rate_t | 0.440 | 0.550 | 3.070 | 0.970 | 32 | 14.084 | 14.237 | 0.235 | 0.017 | 0.105 | 0.054 |
| cohface_10_3 | cohface_10_3.pkl | parh_ossm | external_rate_posterior_mean_t | 0.590 | 0.740 | 4.000 | 0.980 | 32 | 13.860 | 14.237 | 0.230 | 0.015 | 0.105 | 0.054 |
| cohface_10_3 | cohface_10_3.pkl | parh_ossm | external_rate_posterior_mode_t | 0.410 | 0.480 | 2.960 | 0.970 | 32 | 14.378 | 14.237 | 0.240 | 0.019 | 0.105 | 0.054 |
| cohface_10_3 | cohface_10_3.pkl | parh_ossm | final_track_hz | 0.130 | 0.180 | 0.890 | 0.990 | 32 | 14.211 | 14.237 | 0.231 | 0.029 | 0.105 | 0.054 |
| cohface_10_3 | cohface_10_3.pkl | parh_ossm | native_causal_track_hz | 0.190 | 0.230 | 1.360 | 0.990 | 32 | 14.383 | 14.237 | 0.230 | 0.054 | 0.105 | 0.054 |
| cohface_10_3 | cohface_10_3.pkl | parh_ossm | native_smoothed_track_hz | 0.090 | 0.120 | 0.650 | 0.990 | 32 | 14.237 | 14.237 | 0.231 | 0.035 | 0.105 | 0.054 |
| cohface_10_3 | cohface_10_3.pkl | parh_ossm | state_freq_t | 0.720 | 0.830 | 5.230 | 0.970 | 32 | 14.437 | 14.237 | 0.247 | 0.011 | 0.105 | 0.054 |
| cohface_11_0 | cohface_11_0.pkl | parh_ossm | external_output_rate_t | 0.820 | 0.870 | 6.290 | 0.700 | 32 | 13.937 | 13.120 | 0.233 | 0.003 | 0.079 | 0.132 |
| cohface_11_0 | cohface_11_0.pkl | parh_ossm | external_rate_posterior_mean_t | 0.940 | 0.970 | 7.190 | 0.710 | 32 | 14.057 | 13.120 | 0.236 | 0.007 | 0.079 | 0.132 |
| cohface_11_0 | cohface_11_0.pkl | parh_ossm | external_rate_posterior_mode_t | 1.250 | 1.270 | 9.560 | 0.730 | 32 | 14.368 | 13.120 | 0.241 | 0.007 | 0.079 | 0.132 |
| cohface_11_0 | cohface_11_0.pkl | parh_ossm | final_track_hz | 0.160 | 0.190 | 1.240 | 0.870 | 32 | 13.122 | 13.120 | 0.217 | 0.014 | 0.079 | 0.132 |
| cohface_11_0 | cohface_11_0.pkl | parh_ossm | native_causal_track_hz | 0.200 | 0.250 | 1.520 | 0.760 | 32 | 13.119 | 13.120 | 0.218 | 0.047 | 0.079 | 0.132 |
| cohface_11_0 | cohface_11_0.pkl | parh_ossm | native_smoothed_track_hz | 0.140 | 0.170 | 1.050 | 0.910 | 32 | 13.088 | 13.120 | 0.216 | 0.026 | 0.079 | 0.132 |
| cohface_11_0 | cohface_11_0.pkl | parh_ossm | state_freq_t | 0.390 | 0.450 | 3.010 | 0.170 | 32 | 13.238 | 13.120 | 0.224 | 0.019 | 0.079 | 0.132 |
| cohface_11_1 | cohface_11_1.pkl | parh_ossm | external_output_rate_t | 1.710 | 1.850 | 13.950 | 0.830 | 34 | 14.322 | 12.613 | 0.242 | 0.005 | 0.215 | 0.167 |
| cohface_11_1 | cohface_11_1.pkl | parh_ossm | external_rate_posterior_mean_t | 1.910 | 2.030 | 15.560 | 0.790 | 34 | 14.526 | 12.613 | 0.247 | 0.005 | 0.215 | 0.167 |
| cohface_11_1 | cohface_11_1.pkl | parh_ossm | external_rate_posterior_mode_t | 2.510 | 2.610 | 20.300 | 0.700 | 34 | 15.119 | 12.613 | 0.254 | 0.003 | 0.215 | 0.167 |
| cohface_11_1 | cohface_11_1.pkl | parh_ossm | final_track_hz | 0.690 | 0.780 | 5.650 | 0.970 | 34 | 13.302 | 12.613 | 0.223 | 0.024 | 0.215 | 0.167 |
| cohface_11_1 | cohface_11_1.pkl | parh_ossm | native_causal_track_hz | 0.340 | 0.370 | 2.710 | 0.900 | 34 | 12.638 | 12.613 | 0.225 | 0.063 | 0.215 | 0.167 |
| cohface_11_1 | cohface_11_1.pkl | parh_ossm | native_smoothed_track_hz | 0.160 | 0.190 | 1.260 | 0.980 | 34 | 12.696 | 12.613 | 0.222 | 0.044 | 0.215 | 0.167 |
| cohface_11_1 | cohface_11_1.pkl | parh_ossm | state_freq_t | 1.940 | 2.100 | 15.860 | 0.170 | 34 | 14.558 | 12.613 | 0.240 | 0.016 | 0.215 | 0.167 |
| cohface_11_2 | cohface_11_2.pkl | parh_ossm | external_output_rate_t | 0.410 | 0.440 | 2.910 | 0.610 | 32 | 14.738 | 14.323 | 0.246 | 0.002 | 0.046 | 0.171 |
| cohface_11_2 | cohface_11_2.pkl | parh_ossm | external_rate_posterior_mean_t | 0.200 | 0.230 | 1.380 | -0.250 | 32 | 14.482 | 14.323 | 0.241 | 0.000 | 0.046 | 0.171 |
| cohface_11_2 | cohface_11_2.pkl | parh_ossm | external_rate_posterior_mode_t | 0.850 | 0.860 | 5.940 | 0.670 | 32 | 15.172 | 14.323 | 0.253 | 0.001 | 0.046 | 0.171 |
| cohface_11_2 | cohface_11_2.pkl | parh_ossm | final_track_hz | 0.110 | 0.130 | 0.800 | 0.770 | 32 | 14.391 | 14.323 | 0.242 | 0.011 | 0.046 | 0.171 |
| cohface_11_2 | cohface_11_2.pkl | parh_ossm | native_causal_track_hz | 0.130 | 0.150 | 0.900 | 0.560 | 32 | 14.360 | 14.323 | 0.241 | 0.037 | 0.046 | 0.171 |
| cohface_11_2 | cohface_11_2.pkl | parh_ossm | native_smoothed_track_hz | 0.100 | 0.120 | 0.720 | 0.760 | 32 | 14.354 | 14.323 | 0.242 | 0.012 | 0.046 | 0.171 |
| cohface_11_2 | cohface_11_2.pkl | parh_ossm | state_freq_t | 0.400 | 0.440 | 2.800 | -0.010 | 32 | 14.722 | 14.323 | 0.246 | 0.008 | 0.046 | 0.171 |
| cohface_11_3 | cohface_11_3.pkl | parh_ossm | external_output_rate_t | 2.000 | 2.050 | 21.660 | 0.760 | 32 | 11.409 | 9.407 | 0.190 | 0.007 | 0.113 | 0.061 |
| cohface_11_3 | cohface_11_3.pkl | parh_ossm | external_rate_posterior_mean_t | 2.130 | 2.160 | 22.950 | 0.830 | 32 | 11.534 | 9.407 | 0.199 | 0.009 | 0.113 | 0.061 |
| cohface_11_3 | cohface_11_3.pkl | parh_ossm | external_rate_posterior_mode_t | 2.450 | 2.500 | 26.440 | 0.490 | 32 | 11.856 | 9.407 | 0.202 | 0.009 | 0.113 | 0.061 |
| cohface_11_3 | cohface_11_3.pkl | parh_ossm | final_track_hz | 1.670 | 1.760 | 18.110 | 0.510 | 32 | 11.081 | 9.407 | 0.184 | 0.020 | 0.113 | 0.061 |
| cohface_11_3 | cohface_11_3.pkl | parh_ossm | native_causal_track_hz | 1.520 | 1.640 | 16.420 | 0.400 | 32 | 10.904 | 9.407 | 0.182 | 0.050 | 0.113 | 0.061 |
| cohface_11_3 | cohface_11_3.pkl | parh_ossm | native_smoothed_track_hz | 1.510 | 1.610 | 16.260 | 0.510 | 32 | 10.899 | 9.407 | 0.184 | 0.031 | 0.113 | 0.061 |
| cohface_11_3 | cohface_11_3.pkl | parh_ossm | state_freq_t | 1.570 | 1.680 | 17.170 | 0.100 | 32 | 10.980 | 9.407 | 0.184 | 0.012 | 0.113 | 0.061 |
| cohface_12_0 | cohface_12_0.pkl | parh_ossm | external_output_rate_t | 0.980 | 1.170 | 7.610 | 0.960 | 49 | 14.468 | 13.488 | 0.242 | 0.006 | 0.125 | 0.123 |
| cohface_12_0 | cohface_12_0.pkl | parh_ossm | external_rate_posterior_mean_t | 1.080 | 1.290 | 8.390 | 0.960 | 49 | 14.568 | 13.488 | 0.246 | 0.005 | 0.125 | 0.123 |
| cohface_12_0 | cohface_12_0.pkl | parh_ossm | external_rate_posterior_mode_t | 1.360 | 1.420 | 10.340 | 0.960 | 49 | 14.850 | 13.488 | 0.250 | 0.011 | 0.125 | 0.123 |
| cohface_12_0 | cohface_12_0.pkl | parh_ossm | final_track_hz | 1.040 | 1.230 | 8.040 | 0.810 | 49 | 14.526 | 13.488 | 0.248 | 0.019 | 0.125 | 0.123 |
| cohface_12_0 | cohface_12_0.pkl | parh_ossm | native_causal_track_hz | 0.480 | 0.560 | 3.670 | 0.940 | 49 | 13.918 | 13.488 | 0.244 | 0.053 | 0.125 | 0.123 |
| cohface_12_0 | cohface_12_0.pkl | parh_ossm | native_smoothed_track_hz | 1.150 | 1.400 | 8.880 | 0.370 | 49 | 14.614 | 13.488 | 0.248 | 0.036 | 0.125 | 0.123 |
| cohface_12_0 | cohface_12_0.pkl | parh_ossm | state_freq_t | 1.590 | 1.810 | 12.240 | 0.480 | 49 | 15.075 | 13.488 | 0.251 | 0.006 | 0.125 | 0.123 |
| cohface_12_1 | cohface_12_1.pkl | parh_ossm | external_output_rate_t | 0.800 | 1.120 | 6.160 | 0.830 | 32 | 14.865 | 14.227 | 0.249 | 0.011 | 0.148 | 0.081 |
| cohface_12_1 | cohface_12_1.pkl | parh_ossm | external_rate_posterior_mean_t | 0.780 | 1.090 | 5.980 | 0.750 | 32 | 14.737 | 14.227 | 0.243 | 0.011 | 0.148 | 0.081 |
| cohface_12_1 | cohface_12_1.pkl | parh_ossm | external_rate_posterior_mode_t | 1.110 | 1.360 | 8.370 | 0.850 | 32 | 15.336 | 14.227 | 0.260 | 0.015 | 0.148 | 0.081 |
| cohface_12_1 | cohface_12_1.pkl | parh_ossm | final_track_hz | 0.690 | 1.070 | 5.440 | 0.980 | 32 | 14.885 | 14.227 | 0.237 | 0.021 | 0.148 | 0.081 |
| cohface_12_1 | cohface_12_1.pkl | parh_ossm | native_causal_track_hz | 0.440 | 0.650 | 3.440 | 0.990 | 32 | 14.632 | 14.227 | 0.243 | 0.062 | 0.148 | 0.081 |
| cohface_12_1 | cohface_12_1.pkl | parh_ossm | native_smoothed_track_hz | 0.620 | 1.050 | 4.940 | 0.920 | 32 | 14.790 | 14.227 | 0.236 | 0.033 | 0.148 | 0.081 |
| cohface_12_1 | cohface_12_1.pkl | parh_ossm | state_freq_t | 1.040 | 1.610 | 8.180 | -0.130 | 32 | 15.251 | 14.227 | 0.256 | 0.015 | 0.148 | 0.081 |
| cohface_12_2 | cohface_12_2.pkl | parh_ossm | external_output_rate_t | 0.950 | 1.020 | 9.770 | 0.860 | 35 | 10.945 | 9.991 | 0.180 | 0.008 | 0.080 | 0.091 |
| cohface_12_2 | cohface_12_2.pkl | parh_ossm | external_rate_posterior_mean_t | 0.870 | 0.930 | 8.930 | 0.920 | 35 | 10.864 | 9.991 | 0.180 | 0.011 | 0.080 | 0.091 |
| cohface_12_2 | cohface_12_2.pkl | parh_ossm | external_rate_posterior_mode_t | 1.010 | 1.080 | 10.340 | 0.810 | 35 | 11.003 | 9.991 | 0.181 | 0.008 | 0.080 | 0.091 |
| cohface_12_2 | cohface_12_2.pkl | parh_ossm | final_track_hz | 0.170 | 0.230 | 1.800 | 0.970 | 35 | 10.163 | 9.991 | 0.173 | 0.019 | 0.080 | 0.091 |
| cohface_12_2 | cohface_12_2.pkl | parh_ossm | native_causal_track_hz | 0.160 | 0.200 | 1.680 | 0.940 | 35 | 10.045 | 9.991 | 0.176 | 0.043 | 0.080 | 0.091 |
| cohface_12_2 | cohface_12_2.pkl | parh_ossm | native_smoothed_track_hz | 0.100 | 0.130 | 1.040 | 0.970 | 35 | 10.021 | 9.991 | 0.172 | 0.023 | 0.080 | 0.091 |
| cohface_12_2 | cohface_12_2.pkl | parh_ossm | state_freq_t | 1.120 | 1.160 | 11.390 | 0.920 | 35 | 11.111 | 9.991 | 0.181 | 0.012 | 0.080 | 0.091 |
| cohface_12_3 | cohface_12_3.pkl | parh_ossm | external_output_rate_t | 1.140 | 1.220 | 12.700 | 0.970 | 32 | 10.484 | 9.342 | 0.178 | 0.010 | 0.099 | 0.103 |
| cohface_12_3 | cohface_12_3.pkl | parh_ossm | external_rate_posterior_mean_t | 1.630 | 1.690 | 17.920 | 0.970 | 32 | 10.967 | 9.342 | 0.187 | 0.009 | 0.099 | 0.103 |
| cohface_12_3 | cohface_12_3.pkl | parh_ossm | external_rate_posterior_mode_t | 1.220 | 1.290 | 13.460 | 0.970 | 32 | 10.558 | 9.342 | 0.181 | 0.011 | 0.099 | 0.103 |
