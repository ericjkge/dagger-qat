---
slug: on-policy-distillation
status: active
---
# on-policy-distillation

**Mechanism.** Our goal is to use on-policy distillation on the robot-use reasoning model recently trained by Song Han's group (https://arxiv.org/pdf/2611.18562) in order to reduce reasoning latency.

Here is a very rough baseline of the pipeline we plan to carry out for this project.

**Setup.** Our experimental setup will be as follows:
- Teacher: Qwen3.8-27B-PEAR-LIBERO_only ($\pi_{\text{teacher}}$)
- Student: Qwen3.5-9B, Qwen3.5-4B ($\pi_\theta$)
- Infrastructure: PEAR repo (reuse LIBERO environment, prompt format, RL trainer)  

**Baselines.** We will evaluate the naive Qwen3.5-9B and 4B base models using PEAR's exact protocol, with 2000 episodes (40 teasks x 50 initial states). For comparisons, we will use the paper's numbers for the teacher at xhigh (74.4%) and low effort (21.9%).

We will first derive teacher data by running it on all 4 LIBERO suites, keeping only successful episodes (up to around 1000). We will save the full multi-round contexts, including images, reasoning, and actions. 

Next, we want to have an SFT-only baseline to demonstrate that on-policy distillation outperforms it. We will fine-tune each student on the teacher episodes only using standard SFT, then compute the loss on all generated tokens, using the same context format as at inference. We will evaluate results for this model

**Our method.** This is my main contribution and focus on this project. Our on-policy distillation method will utilize the following algorithm:

**Algorithm 1. On-policy distillation**
Input: Student $\pi_{\theta_0}$; frozen teacher $\pi_\text{teacher}$; LIBERO episodes $\mathcal{D}$ (task x initial state), batch size $\mathcal{B}$ , # steps $K$, eval interval $N=10$. 
1. For $k=0, \dots,K-1$:
	2. Rollout: sample $B$ episodes from $\mathcal{D}$ and run $\pi_{\theta_k}$ on each in LIBERO. Record each token sequence $x$, its generated positions $\mathcal{G}(x)$, and $\log \pi_{\theta_k}(x_t \mid x_{<t})$ for $t \in \mathcal{G}(x)$.
	3. Teacher prefill: run one forward pass of $\pi_{\text{teacher}}$ over each $x$, with the same tokens and no generation. This gives $\log \pi_{\text{teacher}}(x_t \mid x_{<t})$ for $t \in \mathcal{G}(x)$.
	4. Advantages: compute $A_t$ as above for every generated token.
	5. Update: take one gradient step on the policy-gradient loss, averaged over all generated tokens in the batch: $$
	     \ell(\theta) = -\frac{1}{\sum_{x}|\mathcal{G}(x)|}\sum_{x}\sum_{t \in \mathcal{G}(x)} A_t ,\log \pi_\theta(x_t \mid x_{<t})
	     $$
	6. Evaluate: if $(k+1) \bmod N = 0$, measure the success rate on validation episodes.
2. Return $\pi_{\theta_K}$.

I want to try using the SFT checkpoint as a student, as well as a untuned student to see whether OPD can perform well without this mid-training step.

Our final evaluation will use LIBERO (2000 episodes) with the same metrics and setup from the baseline evals. To assess generalization ability of the students, we will perform the following (with $\approx$ 10 epidoes per task for each perturbation):
- LIBERO-Pro task perturbation (teacher: 57.4%)
- LIBERO-Plus language and layout perturbations

Our main plot will show our success rates vs. time per episode.