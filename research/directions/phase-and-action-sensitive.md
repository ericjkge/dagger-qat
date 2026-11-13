---
slug: phase-and-action-sensitive
status: active
---
# phase-and-action-sensitive

**Mechanism.** I want to build a 

**Algorithm: Phase-Aware Action-Sensitive Quantization**

**Input:** VLA $A = \psi(f(\text{obs}))$, LIBERO episodes, bits $\{2,3,4,8\}$, budget $\bar b$

1. **Split by phase.** Tag frames approach / grasp / transport / place using gripper flips. Sample equally from each.
2. **Measure gradients.** Estimate $g_j \approx \frac{\partial A}{\partial W_j}$ — how each weight moves the action — using $k$ random-projection backprops.
3. **Score damage.** For each weight, bit-width, and phase: how far would the action move?  
    $s^{(p)}(W_j, b) = \text{mean}_{p}\,\lVert g_j\,(Q_b(W_j) - W_j)\rVert^2$
4. **Take the worst phase.** $s(W_j, b) = \max_p s^{(p)}(W_j, b)$
5. **Allocate bits.** Start at 8 bits. Repeatedly lower whichever weight causes the least damage per bit saved, until the budget is met.

**Output:** bits per weight. Evaluate success per phase.

**Baseline.** We will test the method on 

**Evaluation data.** 