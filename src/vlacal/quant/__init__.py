"""Fake-quant primitives, calibration-statistics capture, SmoothQuant and GPTQ.

Entry point: `quantize_model(model, QuantConfig(...), run_calibration)`.
"""

from vlacal.quant.pipeline import QuantConfig, QuantizedModel, quantize_model

__all__ = ["QuantConfig", "QuantizedModel", "quantize_model"]
