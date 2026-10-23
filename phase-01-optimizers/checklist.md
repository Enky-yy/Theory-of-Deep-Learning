# Phase 1 — The Training Engine: Optimizers (2 weeks)

Goal: every major optimizer explained from its update equation — what it fixes, what it
assumes, what it breaks. You should be able to write Adam's update from memory and
explain every symbol's purpose, then say exactly where it fails.

## Concepts

- Gradient descent family: full-batch GD vs SGD — why the noise in SGD is a *feature*
  (escape saddles, implicit regularization) not just a cost
- Momentum: the heavy-ball derivation; why accumulated velocity navigates ravines;
  Nesterov lookahead and what it anticipates
- The adaptive lineage as one story: AdaGrad (per-coordinate LR from cumulative squared
  grads — dies by monotonically shrinking LR) → RMSProp (decay the accumulator — fixes
  the death) → Adam (momentum on top of RMSProp, bias-corrected)
- Adam term-by-term: m (first moment = smoothed gradient direction), v (second moment =
  per-coordinate scale), m̂/v̂ bias correction and why step 1 needs it (initialization
  at zero), ε's role, default β₁=0.9 / β₂=0.999 and what those timescales mean
- **AdamW**: why L2 regularization and weight decay diverge under adaptive LR — L2
  scales the gradient by λθ *before* the v-normalization, so effectively-decayed
  coordinates get boosted back up; decoupled decay applies θ ← θ − lr·λθ *after*,
  independent of v. This is *the* reason AdamW won.
- LR schedules: step, cosine, and warmup — why Adam on step 1 with full LR is
  dangerous (v underestimated → effective step too large; the bias correction
  doesn't fully save it at scale)
- Gradient clipping: by value (elementwise, distorts direction) vs by norm (preserves
  direction, rescales) — and which one RNN training needs and why (Phase 4 foreshadow)
- Loss landscapes: saddle points as the typical obstacle in high-dim (Phase 0),
  sharp vs flat minima, and the honest caveat that the flatness-generalization
  story is more contested than textbook treatments admit
- Batch size ↔ LR scaling heuristics; large-batch generalization gap; why
  "linear scaling rule" works for SGD and where it stops working
- Mixed precision: what fp16/bf16 breaks (small gradients underflow), gradient
  scaling as the fix

## Derivations (on paper)

- [ ] Adam from scratch: write all five equations (m, v, m̂, v̂, θ update) from memory,
      then derive E[m_t] = (1−β₁ᵗ)·ḡ and hence the 1/(1−β₁ᵗ) correction
- [ ] Momentum vs heavy-ball: show the β-recurrence as an exponential moving average
      of gradients, and that its effective horizon is ~1/(1−β) steps
- [ ] L2 vs weight decay under Adam: write both update rules side by side and show
      they differ by a factor of 1/√v — then say in words when the difference vanishes
- [ ] AdaGrad's accumulator is monotone non-decreasing → LR → 0; derive the rate of
      decay on a coordinate with constant gradient (it goes as 1/√t)

## Misconceptions to kill

- [ ] "Adam adapts the learning rate" — it adapts a *per-coordinate preconditioner*;
      the scalar lr is separate. Conflating the two hides why warmup matters.
- [ ] "Weight decay is just L2 with another name" — true for SGD, false for Adam.
- [ ] "Gradient clipping fixes vanishing gradients" — no: clipping bounds the *large*
      ones. It treats explosion, ignores vanishing, and is why Phase 4's RNN problem
      is only half-solvable this way.
- [ ] "Higher training accuracy with Adam means it's the better optimizer" — compare
      at equal compute and equal tuning budget; part of Adam's reputation is
      cheap-to-tune, not better-optima.

## Thought experiments

- [ ] A loss surface is a narrow ravine: axis 1 steep, axis 2 flat, minimum along the
      diagonal. Sketch 20 steps of vanilla GD vs momentum on it. Where does each go wrong?
- [ ] β₂ = 0.999 means v remembers ~1000 steps. The loss landscape's curvature changes
      dramatically at epoch 5. What is v *doing* during that transition, and what does
      the model's learning look like in that window?
- [ ] Training a transformer with Adam, LR warmup of 2000 steps, peak 3e-4. Someone
      removes the warmup. Predict the loss curve, then explain the mechanism twice:
      once via v, once via the effective-step-size formula.

## Cold quiz
1. Write Adam's full update rule from memory; explain every symbol in one phrase each.
2. Why bias correction? What exactly is biased, and what would happen without it at step 1?
3. L2 under Adam vs decoupled weight decay: the mechanism of divergence, in three sentences.
4. Why does transformer training need warmup but a CIFAR ResNet with SGD mostly doesn't?
5. Clip-by-norm vs clip-by-value: mechanism, and which suits exploding-gradient RNNs.
6. Give one concrete situation where SGD+momentum beats Adam, and the mechanism behind it.
