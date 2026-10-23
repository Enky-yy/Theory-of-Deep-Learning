# Phase 4 — RNNs: Recurrence, Hidden State, BPTT (2 weeks)

Goal: the recurrent mechanism itself — hidden state as memory, backprop through time
as ordinary backprop on an unrolled graph, and the vanishing-gradient problem
*derived*, not hand-waved. This phase makes Phase 5's gates feel inevitable.

## Concepts

- **Why recurrence**: fixed windows fail on long-range dependencies; the alternative
  is a computation whose step depends on its own previous step — parameter sharing
  *across time* (the same W applied at every timestep: what it buys in parameters,
  what it imposes as a constraint)
- The vanilla RNN equations, said precisely:
  h_t = tanh(W_hh·h_{t−1} + W_xh·x_t + b),  y_t = W_hy·h_t (+ b')
  — every term's role; state dimension; the fact that W_hh is the *same matrix* at
  every step
- **Hidden state as memory**: h_t as a (lossy, fixed-size) summary of x_1..x_t;
  the information bottleneck view — fixed d floats must encode all history; why
  this is both the RNN's power and its limit
- **BPTT**: unrolling the RNN into a deep feedforward net with *tied weights*;
  ordinary backprop on that graph; the ∂h_t/∂h_{t−1} = W_hhᵀ·diag(f'(z_t)) term
  repeated across timesteps
- **Vanishing/exploding gradients, derived**: the gradient from loss at time T back
  to time k is a *product of (T−k) Jacobians*; if ‖W_hh‖ < 1 the signal decays
  geometrically — the model provably cannot learn dependencies longer than
  ~O(1/(1−ρ)) steps, where ρ is the effective spectral radius; > 1 explodes
- Consequences you can predict: long-range dependencies fail *silently* (training
  loss looks fine); gradient clipping (norm-based, from Phase 1) treats the symptom;
  careful initialization (orthogonal W_hh, identity-ish) delays but doesn't cure it
- Truncated BPTT: the pragmatic compromise and exactly what it gives up
- Sampling vs teacher forcing: training/inference mismatch (exposure bias)
- Bidirectional RNNs: the two hidden states concatenated — what it needs (full
  sequence available) and where it's illegal (online/causal tasks)
- Deep (stacked) RNNs: layer-wise vs time-wise depth; why stacking compounds
  the vanishing problem
- The many-to-one / one-to-many / many-to-many taxonomy — with the NLP tasks each
  maps to (classification, captioning, tagging, translation)
- Variable-length sequences: packing/masking, why padding tokens must not pollute
  the hidden state

## Derivations (on paper)

- [ ] Unroll a 2-timestep RNN, write every Jacobian of L w.r.t. W_hh explicitly;
      identify the repeated structure
- [ ] The full ∂L/∂h_k = Σ_t (∏_{i=k+1..t} W_hhᵀ diag(f'(z_i))) (∂L/∂y_t ∂y_t/∂h_t)
      chain — write it, then bound its norm to show vanishing
- [ ] Construct the numerical case: ‖W_hh‖₂ = 0.9, 100 timesteps → signal factor
      0.9¹⁰⁰ ≈ 2.6×10⁻⁵. Do the arithmetic, feel the number
- [ ] Why tanh helps sigmoid but not the fundamental problem (|f'| ≤ 1 either way;
      the spectral radius of W_hh dominates)

## Misconceptions to kill

- [ ] "RNNs remember everything until the window fills" — they have no window;
      they have a *decaying* memory whose horizon is set by the gradient-flow
      math above. The forgetting is gradual and structural, not a cutoff.
- [ ] "Vanishing gradients means the loss stops improving" — no: the model learns
      short dependencies fine and *appears* trained; the failure shows only on
      examples requiring long memory.
- [ ] "BPTT is a special algorithm" — it's ordinary backprop on the unrolled graph
      with tied parameters. Anyone saying "special" hasn't unrolled it.
- [ ] "Hidden state is like a database/lookup" — it's a *lossy fixed-size summary*,
      continuously overwritten, never addressed. This is why copying tasks break it.

## Thought experiments

- [ ] Sequence task: classify whether the *first* token is in the vocabulary of the
      100th token (copy-like task). Predict train behavior for LSTM-vs-RNN *before*
      seeing Phase 5 results — write the prediction down now, verify later.
- [ ] Hidden state dimension 256, sequence length 1000, vocabulary of characters.
      Estimate the information bottleneck: what's the theoretical minimum bits to
      store the full history? What does that comparison imply?
- [ ] Truncated BPTT with window 50 on a task needing dependency length 200.
      What can the model learn, mechanistically? Where does the signal die?

## Cold quiz
1. Write the vanilla RNN equations; label each weight's role; say what "tied across
   time" means for parameter count and for the learning constraint.
2. Derive the gradient flow from L_t back to h_k; state the vanishing condition
   in terms of the spectral radius; give the order-of-magnitude horizon for ρ=0.9.
3. Hidden state: one precise sentence on what it *is* (bottleneck, lossy summary,
   continuously overwritten) — then two tasks that must fail because of it.
4. Teacher forcing: the mechanism of exposure bias, and how it manifests at
   inference time.
5. When is a bidirectional RNN illegal? Give the concrete deployment reason.
6. Truncated BPTT: what does it trade away? Construct a task where it provably fails.
