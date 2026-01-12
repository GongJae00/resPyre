
import json
from core.pipeline.wrapped_method import OscillatorWrappedMethod
from components.models.core.base import OscillatorParams

def test():
    base_cfg = {
        "name": "test",
        "methods": [
            {
                "name": "of_farneback__kfstd",
                "params": {
                    "oscillator": {
                        "qx": 0.0005,
                        "no_autotune": True
                    }
                }
            }
        ]
    }
    method_cfg = base_cfg['methods'][0]
    m = OscillatorWrappedMethod(method_cfg)
    print(f"Initial qx: {m.osc_head.params.qx}")
    
    # Simulate what happens in run_optuna.py
    params = {"oscillator.qx": 0.0001}
    # In run_optuna, we do:
    # _set_nested(cfg['methods'][0], f"params.{path}", float(val))
    # which effectively updates method_cfg['params']['oscillator']['qx']
    
    # Let's see if we can just re-init or if it needs to be passed correctly.
    # The wrapped method initializes the head in __init__.
    
test()
