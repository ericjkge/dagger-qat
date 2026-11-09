# GPTQ numerical reference

`gptq-reference.npz` contains synthetic weights, two variable-length input
sequences, and quantized outputs produced by the original `GPTQ.fasterquant`
implementation from IST-DASLab/gptq commit
`2d65066eeb06a5c9ff5184d8cebdf33662c67faf` (`gptq.py` and `quant.py`).

Settings: W4, asymmetric per-row grids, group size 8, block size 8, damping 0.01,
natural column order, dynamic groups, no MSE grid search. Both FP32 and BF16 input
and weight paths are included. Only the reference's unconditional
`torch.cuda.synchronize()` was replaced with a no-op to run it on CPU. The
reference computations and quantizer were otherwise unchanged.

Synthetic inputs use `torch.manual_seed(2026)`: `weights = randn(6, 16)`, with
row 0 replaced by `abs(row) + 0.2`; then `first = randn(1, 31, 16)` and
`second = randn(1, 23, 16)`. In each input, channel 1 is replaced by
`0.8 * channel_0 + 0.1 * channel_1`, and channel 15 is zeroed. The fixture covers
correlated inputs, a dead Hessian column, nonzero-only weight ranges and BF16
rounding. Expected outputs are generated from the reference, not our port.

Reference source: https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf

The upstream Apache-2.0 license is retained in `licenses/gptq.txt`.
