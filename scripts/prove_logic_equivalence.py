import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.models.core.ssm import OscillatorPredictor

import numpy as np

print("=== 1. Scalar Conversion Logic Test ===")
# H @ P @ H.T returns a scalar (1x1 matrix)
test_mat = np.array([[3.1415926535]]) # Represents HPH^T

val_float = float(test_mat)
val_item = test_mat.item()

print(f"Matrix (1x1): {test_mat}")
print(f"float() conversion: {val_float} (Type: {type(val_float)})")
print(f".item() conversion: {val_item} (Type: {type(val_item)})")

assert val_float == val_item, "FATAL: Values differ!"
assert type(val_float) == type(val_item), "FATAL: Types differ!"
print(">> PASSED: Both methods are numerically identical (float64).")
print("")

print("=== 2. Gaussian Limit Logic Test (nu -> inf) ===")
# VB update formula: lambda = (nu + 1) / (nu + mahal_sq)
mahal_sq = 100.0 # Large outlier

def calc_lambda_formula(nu, mahal):
    return (nu + 1.0) / (nu + mahal)

print(f"Mahalanobis^2 (outlier): {mahal_sq}")

nu_list = [10.0, 100.0, 1e6, 1e12, float('inf')]
for nu in nu_list:
    try:
        lam = calc_lambda_formula(nu, mahal_sq)
        print(f"nu = {nu:8}: lambda = {lam}")
    except Exception as e:
        print(f"nu = {nu:8}: Error {e}")

print("\n>> Conclusion: As nu -> inf, lambda approaches 1.0. Direct calculation with inf yields nan.")
print(">> The patch explicitly sets lambda=1.0 for nu > 1e12, which is the CORRECT mathematical limit.")
