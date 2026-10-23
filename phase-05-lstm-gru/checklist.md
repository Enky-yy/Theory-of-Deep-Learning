# Phase 5 — LSTM & GRU: Gated Memory (1.5 weeks)

Goal: every gate's equation, purpose, and failure mode — plus the single structural
insight that fixes Phase 4: an **additive** cell-state path whose gradient doesn't
multiply Jacobians over time. You should be able to write all LSTM and GRU equations
from memory and explain each term's role in one phrase.

## Concepts

- **Cell state vs hidden state — the central distinction**:
  - Cell state c_t = the *long-term memory highway*: additive updates only
    (c_t = f⊙c_{t−1} + i⊙g), so gradients flow through time with only multiplicative
    *scalars* (the forget gate), not full matrices. Gradient ≈ product of f-values,
    which can be ≈ 1 → dependencies survive hundreds of steps.
  - Hidden state h_t = the *working memory / output*: filtered view of c_t through
    tanh, gated by output. What the next layer or prediction actually sees.
  - The conveyor-belt picture: information rides c_t untouched unless gates act on it.
- **The four gates, term by term** (t = tanh candidate, f/i/o = sigmoid gates):
  - Forget gate f = σ(W_f·[h_{t−1}, x_t] + b_f): what to erase from long-term memory.
    Bias-init trick: b_f > 0 starts with "remember everything" — empirically important.
  - Input gate i: what new information to admit; candidate g = tanh(W_g·[h,x]+b_g):
    the proposed content itself.
  - Cell update c_t = f⊙c_{t−1} + i⊙g: the additive path — *the* reason LSTM works.
  - Output gate o: what of the (filtered) cell state to expose as h_t.
  - h_t = o ⊙ tanh(c_t)
- Why sigmoid for gates (0..1 = open/close) vs tanh for content (−1..1, zero-centered);
  the 100-step gradient factor becomes Π f_j, scalars ≈ 1 possible — contrast with
  Phase 4's Π Wᵀdiag(·)
- Parameter count: 4 × (h(x+h+1)) vs vanilla RNN's 2 × — and where that budget goes
- **GRU**: update gate z merges forget+input (z⊙h̃ + (1−z)⊙h_{t−1}), reset gate r
  gates the *candidate's* dependence on past; no separate cell state; fewer params;
  comparable performance (Greff et al. 2015: the details that matter are forget-gate
  bias, output gating, not the cell-vs-hidden distinction itself)
- When LSTM vs GRU: the honest answer — mostly interchangeable; LSTM slightly better
  on long sequences/large data in the original studies; GRU cheaper. Modern practice:
  both displaced by attention for most tasks (Phases 7–8) — but their gate intuition
  lives on in every "gating" mechanism since (attention gates, GLU activations,
  transformer FFN variants)
- Peepholes (cell→gate connections); coupled forget/input gates; the variant zoo
  and why the original paper's exact form isn't sacred
- Deep/bidirectional LSTMs; packing for variable length (from Phase 4)

## Derivations (on paper)

- [ ] All LSTM equations from memory, twice — forward, then with the gradient path
      of ∂c_t/∂c_{t−1} = f_t (+ terms through gates) written out; show why the
      dominant path is scalar-multiplicative
- [ ] The "constant error carousel": why ‖∂L/∂c‖ can stay constant over 100 steps
      when f ≈ 1 — compare numerically to the RNN's 0.9¹⁰⁰ from Phase 4
- [ ] GRU from LSTM: show how z and (1−z) as paired weights collapse forget+input
      into one gate; write both update rules side by side
- [ ] Parameter count arithmetic for h=256, input=128: vanilla RNN vs LSTM vs GRU

## Misconceptions to kill

- [ ] "LSTM remembers long-term dependencies automatically" — it *can* (the highway
      exists); whether it *learns* to keep f≈1 depends on the task's loss gradient
      actually rewarding it. The gate is a possibility, not a guarantee.
- [ ] "The hidden state is the LSTM's memory" — the cell state is; h is a
      filtered projection of it. This distinction is the whole phase.
- [ ] "Gates are hard switches" — they're continuous [0,1] weights; gradients flow
      *through* them; they're soft, learned, per-dimension, per-timestep.
- [ ] "LSTM solved vanishing gradients" — it solved the *matrix-Jacobian* product
      path. Gate-sigmoid saturation (f→0 everywhere) can still sever the highway;
      and the input path through W_g still has matrix Jacobians (bounded by tanh').
- [ ] "GRU is just a smaller LSTM" — different structure: no separate cell state,
      reset gate is unique to GRU; the update is a *convex combination* with the
      previous state, which has its own gradient character.

## Thought experiments

- [ ] Copy-task of length 500: vanilla RNN (expected fail), LSTM (expected pass).
      Mechanistically explain *both outcomes* through the equations — why does the
      LSTM gradient survive?
- [ ] Forget-gate bias set to −5 (start with "forget everything"): predict training
      behavior on a task needing long memory. What does the loss curve look like,
      and when does it recover, if ever?
- [ ] An LSTM's output gate is forced open (o ≡ 1). Name one task that gets easier,
      one that gets harder, with mechanisms.
- [ ] Design a task where GRU beats LSTM (hint: the reset gate's role in *dropping*
      irrelevant past before computing the candidate).

## Cold quiz
1. Write all LSTM equations from memory; label each gate's purpose in ≤5 words;
      say why gates are sigmoid and content is tanh.
2. Cell state vs hidden state: the structural difference, the gradient-flow
      consequence, and the "who sees it" consequence (next step vs next layer/output).
3. Derive ∂c_t/∂c_{t−1} and explain the "constant error carousel" in two sentences,
      contrasting with the Phase-4 RNN product.
4. GRU vs LSTM: the equations that differ, the parameter difference, and the honest
      empirical answer on which is better.
5. The forget-gate bias initialization trick: what and why.
6. Name the two ways an LSTM still suffers vanishing gradients despite the design.
