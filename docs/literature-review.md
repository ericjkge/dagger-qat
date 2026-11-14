# Literature Review

Papers we have looked at so far, grouped by topic. Each entry has a link and a
short high-level summary.

## Imitation learning

### DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (Ross, Gordon & Bagnell, 2011)
[arXiv:1011.0686](https://arxiv.org/abs/1011.0686)

Behavior cloning trains only on expert states, so small mistakes push the learner into states it never saw, and errors compound over the horizon. DAgger fixes this iteratively: roll out the current learner, have the expert label the states the learner actually visits, aggregate them into the dataset, and retrain. This gives regret guarantees that grow linearly rather than quadratically in the horizon, and its outer loop is the template for our on-policy calibration.

## VLA models

### π0.5: a Vision-Language-Action Model with Open-World Generalization (Physical Intelligence, 2025)
[arXiv:2504.16054](https://arxiv.org/abs/2504.16054)

π0.5 builds on a PaliGemma VLM (SigLIP vision encoder plus Gemma 2B) and adds a flow-matching/diffusion action expert trained from scratch. Pre-training trains only the VLM on robot data (single frames plus robot state), multimodal web data, and subtask data (goals and bounding boxes). Post-training trains the VLM and action expert together on robot data, web data, and verbal instructions, which lets the model generalize to new homes.

### OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (Kim, Finn & Liang, 2025)
[arXiv:2502.19645](https://arxiv.org/abs/2502.19645)

OFT is a fine-tuning recipe for OpenVLA. It replaces autoregressive token decoding with parallel decoding of action chunks, continuous actions, and an L1 regression objective. It reaches state-of-the-art LIBERO success with much faster inference, and it is the base policy we quantize in our experiments.

## Benchmarks

### LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning (Liu et al., 2023)
[arXiv:2306.03310](https://arxiv.org/abs/2306.03310)

LIBERO is a simulated manipulation benchmark built on robosuite (MuJoCo). It has four suites of 10 tasks each: Spatial (same objects, different layouts), Object (different objects), Goal (same scene, different goals), and Long (multi-step, long-horizon). It has become the standard closed-loop testbed for VLA policies and their quantized variants, and LIBERO-Long is our main evaluation suite.

### LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models (Fei et al., 2025)
[arXiv:2510.13626](https://arxiv.org/abs/2510.13626)

LIBERO-Plus keeps the same 40 LIBERO tasks and adds controlled perturbations along 7 axes: camera viewpoint, robot initial state, language instruction, lighting, background texture, sensor noise, and object layout. Across many VLAs, success drops sharply under modest perturbations, with camera viewpoint and initial state being the most damaging, and models often ignore the language instruction. It is our candidate benchmark for measuring whether on-policy quantization improves OOD robustness.

## VLA quantization

### QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization (ICLR 2026)
[arXiv:2602.03782](https://arxiv.org/abs/2602.03782)

QVLA treats VLA quantization as a budgeted bit-allocation problem: each output channel (row of a linear weight matrix) gets a bit width in {0, 2, 4, 8, 16} under an average-bit budget, where 0 bits means pruning. Channel sensitivity is estimated as quantization noise (how much the channel's output changes) times amplification (the gradient of the action output with respect to that channel), and the most important channels are checked with full forward passes. Bits are then allocated with greedy demotion, and the paper reports near-lossless OpenVLA-OFT performance on LIBERO at roughly 4 bits on average.

### VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models (2026)
[arXiv:2609.25376](https://arxiv.org/abs/2609.25376)

VLAQuantBench evaluates standard PTQ methods on VLAs in closed-loop LIBERO rollouts instead of with offline proxy metrics. It shows that quantization scope, numerical format, calibration, and layer interactions all affect task success. Weight-only W4/W8 stays close to full precision while W4A4 can collapse to 0%, and a small OFT output projection can dominate the failures.

### QbD-VLA: Quantize by Doing: Task-Guided Mixed-Precision Quantization of Vision-Language-Action Models (2026)
[arXiv:2610.38661](https://arxiv.org/abs/2610.38661) · [code](https://github.com/jiuyixu25/QbD-VLA)

QbD-VLA chooses per-layer weight and activation bit widths from closed-loop task success in rollouts, instead of from sensitivity scores on calibration data. It uses calibration-free round-to-nearest quantization and trains no policy. It reaches about W4 / under-A6 on average across four VLAs and all LIBERO suites while beating sensitivity-based allocations by a wide margin, but rollout-based search is expensive. That cost is what our "more efficient rollout-based quantization" idea targets.

### QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models (Zhang et al., CVPR 2026)
[arXiv:2602.20309](https://arxiv.org/abs/2602.20309)

QuantVLA attributes the fragility of a VLA's DiT action head to quantization-induced scale drift, which changes the attention temperature (the effective softmax logit scale) and the residual-stream energy. It quantizes the linear layers of the language backbone and the action head but keeps attention projections in floating point. It then corrects the drift with two training-free calibrations folded into the dequantization scales: per-head attention temperature matching, and per-layer output-head balancing at the residual interface.

### CHASE-VLA: Post-Training Quantization Framework for Vision-Language-Action Models with Chunk-Aware Scale Estimation (Hyun et al., 2026)
[arXiv:2610.02666](https://arxiv.org/abs/2610.02666)

CHASE-VLA observes that activation ranges in a diffusion action expert shift with both the motion context and the denoising step, so a single static scale fits poorly. It sets activation scales conditioned on the action chunks the policy predicted earlier and on the denoising-step group, with the policy kept frozen. The paper reports W4A4 for the action expert's MLP and attention projections on π0.5 at about FP16-level LIBERO success (97.3%).

### HoloQ-VLA (also titled Ω-QVLA): Uniform W4A4 Quantization of Vision-Language-Action Models (2026)
[arXiv:2605.28803](https://arxiv.org/abs/2605.28803)

HoloQ-VLA targets weight and activation outliers in both the language backbone and the diffusion action head, along with dynamic-range drift across denoising steps. It combines a composite SVD·Hadamard rotation to remove outliers with per-step activation scaling in the DiT head that absorbs the drift. This yields training-free, uniform W4A4 quantization of the whole VLA without mixed precision, and the paper reports about 98% on LIBERO with π0.5.

### ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models (Akbari et al., 2026)
[arXiv:2605.24011](https://arxiv.org/abs/2605.24011)

ActQuant is a two-stage mixed-precision PTQ method for VLA backbones. First, an action-aware HSIC sensitivity score assigns each weight matrix a single bit width under a budget, which keeps every matmul on one dense integer kernel. Second, per-block scales inside each matrix are tuned with element-wise weights from an "Action-Mixed Fisher," and the paper reports 2.5 bits per weight at 90.1% success on OpenVLA-OFT (a 5.3× smaller backbone).

### QAIL: Quantization-Aware Imitation-Learning for Resource-Efficient Robotic Control (Park et al., 2024)
[arXiv:2412.01034](https://arxiv.org/abs/2412.01034)

QAIL fine-tunes imitation-learning policies with quantization in the loop. Naive QAT plus behavior cloning loses accuracy on sequential tasks, so it adds a Quantization-Robust Behavior Cloning loss that matches the quantized policy's action distribution to the full-precision policy's. It reports large speed and energy gains at 4-bit on LIBERO manipulation and on autonomous driving, and it is the closest prior work to distillation-based, DAgger-style QAT.

## LLM / general quantization methods

### RTN (round-to-nearest)
Baseline, not a paper. Round each weight (and optionally activation) to the nearest grid point without calibration.

### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (Dettmers et al., 2022)
[arXiv:2208.07339](https://arxiv.org/abs/2208.07339)

At scale, transformers develop a few activation feature dimensions with huge outliers that break naive 8-bit quantization. LLM.int8() uses vector-wise quantization for most of the matmul and moves the outlier channels into a separate FP16 matmul. This gives W8A8 inference without accuracy loss and needs no calibration.

### SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models (Xiao et al., 2022)
[arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

Activations have extreme outliers in a few channels while weights are relatively smooth, so uniform activation quantization squashes most values. SmoothQuant applies a mathematically equivalent per-channel rescaling that divides activations and multiplies weights, which moves the quantization difficulty from activations to weights. Scales come from activation maxima on calibration data, and the method enables W8A8 with real speedups.

### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)
[arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

GPTQ quantizes a layer's weights one input column at a time. After each step it updates the remaining unquantized weights to absorb the error, using the inverse Hessian computed from calibration activations (an Optimal Brain Quantization approximation). It scales to 175B-parameter models in a few GPU hours at 3–4 bits, and it is the inner-loop quantizer in our on-policy calibration experiments.

### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (Lin et al., 2023)
[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

A small fraction of weight channels matter most, and they are identified by activation magnitude, not weight magnitude. AWQ scales up these salient channels before rounding, with a per-channel scale searched on calibration data, so their relative quantization error shrinks without any mixed precision. It is a robust, hardware-friendly W4 method and one of our main baselines.

### SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression (Dettmers et al., 2023)
[arXiv:2306.03078](https://arxiv.org/abs/2306.03078)

SpQR extends GPTQ-style quantization by finding the small set of outlier weights that cause large errors and keeping them in high precision in a sparse format. It also uses small groups with second-level quantized statistics. This makes 3–4-bit compression nearly lossless.

### QuIP: 2-Bit Quantization of Large Language Models With Guarantees (Chee et al., 2023)
[arXiv:2307.13304](https://arxiv.org/abs/2307.13304)

QuIP argues that quantization works best when weights and Hessians are incoherent, meaning no single entry is unusually large. It applies random orthogonal pre- and post-processing to make them incoherent, then uses an adaptive (LDLQ, GPTQ-like) rounding. This gives the first workable 2-bit LLMs along with theoretical guarantees.

### QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks (Tseng et al., 2024)
[arXiv:2402.04396](https://arxiv.org/abs/2402.04396)

QuIP# replaces QuIP's random rotations with faster randomized Hadamard transforms. It quantizes groups of weights jointly with E8-lattice vector codebooks and adds fine-tuning. This pushes the state of the art for 2–4-bit weight-only quantization.

### QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (Ashkboos et al., 2024)
[arXiv:2404.00456](https://arxiv.org/abs/2404.00456)

QuaRot rotates the hidden state with Hadamard matrices in a way that leaves the model's output unchanged but removes activation outliers. With outliers gone, weights, activations, and the KV cache can all go to 4 bits. It gives end-to-end W4A4 inference with small accuracy loss.

### SpinQuant: LLM Quantization with Learned Rotations (Liu et al., 2024)
[arXiv:2405.16406](https://arxiv.org/abs/2405.16406)

Different random rotations give very different quantized accuracy. SpinQuant learns the rotation matrices with Cayley optimization on a small calibration set. This narrows the W4A4 gap to full precision beyond QuaRot.

### OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models (Shao et al., 2023)
[arXiv:2308.13137](https://arxiv.org/abs/2308.13137)

OmniQuant keeps the PTQ setting but learns a few quantization parameters by gradient descent, block by block. It learns weight clipping thresholds (LWC) and SmoothQuant-style equivalent transformations for activations (LET). This light training covers weight-only and weight-activation settings and is a trainable reconstruction baseline for us.

### HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision (Dong et al., 2019)
[arXiv:1905.03696](https://arxiv.org/abs/1905.03696)

HAWQ measures each layer's sensitivity by the top Hessian eigenvalue of the loss with respect to its weights. Sensitive layers get higher bit widths, and a Hessian-based ordering sets the order of quantization-aware fine-tuning. It is a classic Hessian-driven mixed-precision method that QVLA-style allocation builds on.

### QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

QLoRA freezes a 4-bit (NF4) quantized base model and backpropagates through it into low-rank LoRA adapters. It adds double quantization and paged optimizers to save memory. This makes fine-tuning of 65B-parameter models fit on a single 48 GB GPU, and it is a reference point for training with quantization in the loop.
