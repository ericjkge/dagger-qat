---
slug: on-policy-distillation
status: active
---
# on-policy-distillation

**Mechanism.** Our goal is to distill the robot-use reasoning modelrecently trained by Song Han's group (https://arxiv.org/pdf/2611.18562) in order to reduce reasoning latency.

Here is a very rough baseline of the pipeline we plan to carry out for this project.

Setup:

- Teacher: Qwen3.8-27B-PEAR-LIBERO_only (final checkpoint is mainly for real-world deployment and has no paper results)
    
- Student: Qwen3.5-9B (no smaller/open models between this and 3.8), Qwen3.5-4B (second size)
    
- Infrastructure: PEAR repo (reuse LIBERO env, prompt format, RL trainer)
    

  

Step 1: Baselines

- Models: the Qwen3.5-9B and 4B base models. Use the paper’s numbers for the teacher at xhigh (74.4%) and low effort (21.9%).
    
- Evaluation: PEAR’s protocol, 2,000 episodes (40 tasks × 50 initial states).
    

  

Step 2: Teacher data

- Run the teacher on all 4 LIBERO suites.
    
- Keep only successful episodes, up to about 1,000.
    
- Save the full multi-round contexts: images, reasoning, and actions.
    

  

Step 3: SFT

- Fine-tune each student on the teacher episodes.
    
- Compute the loss on all generated tokens, using the same context format as at inference.
    
- Evaluate the result as the “SFT only” model.
    

  

Step 4: On-policy distillation

- Start from SFT checkpoint
    

- Each step:
    

1. The student runs LIBERO episodes.
    
2. The teacher prefills those episodes and gives a log-prob for each student token.
    
3. The per-token advantage is reverse KL loss, with no task reward.
    
4. Do one update.
    

- Evaluate every N=10 steps.
    

  

Step 5: Final evaluation

- LIBERO (2,000 episodes): all metrics from Step 1.
    
- Generalization (students only; about 10 episodes per task for each perturbation):
    

- LIBERO-Pro task perturbation (teacher: 57.4%)
    
- LIBERO-Plus language and layout perturbations
    

- Main plot: success rate vs. time per episode
    