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

**Our method.** Our on-policy distillation method will utilize the following algorithm:



Step 4: On-policy distillation
- Start from SFT checkpoint
- Each step:
	- The student runs LIBERO episodes.
	- The teacher prefills those episodes and gives a log-prob for each student token.
	- The per-token advantage is reverse KL loss, with no task reward.
	- Do one update.
- Evaluate every N=10 steps.
$$
\mathrm{KL}\left(\pi_\theta \,\|\, \pi_{\text{teacher}}\right) = \mathbb{E}_{x \sim \pi_\theta}\left[\log \pi_\theta(x_{t+1} \mid x_{1..t}) - \log \pi_{\text{teacher}}(x_{t+1} \mid x_{1..t})\right] $$

Step 5: Final evaluation
- LIBERO (2,000 episodes): all metrics from Step 1.
- Generalization (students only; about 10 episodes per task for each perturbation):
	- LIBERO-Pro task perturbation (teacher: 57.4%)
	- LIBERO-Plus language and layout perturbations
- Main plot: success rate vs. time per episode