# Phase 0 notes — The mathematical language of deep learning

The minimum math to hold *fluently* before Phase 1. Written for someone at an
intermediate DL level: less "what is a vector," more "why this identity is load-bearing."

## 1. Matrix calculus essentials

- **Shapes rule:** if `y = f(X)` with `X ∈ R^{m×n}`, then `∂y/∂X ∈ R^{m×n}` — the
  gradient has the same shape as the variable. In PyTorch, `X.grad` mirrors `X` exactly
  because of this convention.
- **Chain rule, vector form:** `z = f(y), y = g(x) ⟹ ∂z/∂x = (∂z/∂y)(∂y/∂x)`.
  Deep learning's entire training algorithm is this line, applied over a graph, in reverse.
- **Backprop = reverse-mode autodiff:** one forward pass + one reverse sweep computes
  *all* partial derivatives simultaneously. Cost ≈ 2–3× the forward pass, regardless of
  parameter count — this is why per-parameter gradient cost is O(1), and why training
  billion-parameter models is tractable at all.
- **Gradients to know cold:**
  - `∂(wᵀx + b)/∂w = x`, `∂(wᵀx + b)/∂x = w`
  - `∂σ(z)/∂z = σ(z)(1 − σ(z))` — note it's maximal at z=0 (σ=0.5) and vanishes at
    the saturating ends. This single fact predicts sigmoid's death in deep stacks.
  - `∂CE/∂z = ŷ − y` (softmax + cross-entropy collapse) — the most-used identity in DL
  - `∂‖y − ŷ‖²/∂ŷ = 2(ŷ − y)`
- **Vanishing/exploding gradients = repeated Jacobian products.** Gradient through a
  depth-L network multiplies L Jacobians. If each has spectral norm < 1, the product
  decays geometrically; > 1, explodes. This is *the* reason RNNs forget (Phase 4),
  and motivates residual connections and normalization (Phases 2, 7). Whenever you meet
  a stabilizing trick later, ask: "which Jacobian does this control?"

## 2. Linear algebra intuition checklist

- Matrix = linear map. Eigenvectors = directions the map only scales (by eigenvalues).
  **SVD:** every matrix = rotate → scale (by singular values) → rotate. Singular values
  are the true per-axis stretch factors, even for non-square or singular matrices.
- **Norms:** Frobenius = elementwise L2 of all entries; spectral norm = largest singular
  value = maximum stretch of any unit vector. Weight decay penalizes the Frobenius norm;
  stability/Lipschitz arguments (GANs, Phase 10) use the spectral norm. Different tools,
  different questions.
- **Dot product ≈ alignment.** `qᵀk` measures alignment between q and k. Scaling by
  `√d_k` in attention keeps logits in softmax's sensitive range as dimension grows —
  you'll derive this precisely in Phase 7, but the seed is: dot products of random
  d-dimensional vectors have variance ∝ d.
- **PSD matrices:** `xᵀAx ≥ 0`. Covariance matrices are PSD; `WᵀW` is PSD. Why loss
  landscapes are analyzed through Hessian eigenvalues (negative eigenvalue = saddle,
  not minimum).

## 3. Probability & information theory (the language of loss functions)

- **Entropy:** `H(p) = −Σ p log p` — expected surprisal. A fair 6-sided die: ln 6 nats.
  Certainty: 0.
- **Cross-entropy:** `H(p, q) = −Σ p log q` — expected code length using q when truth
  is p. `H(p,q) = H(p) + KL(p‖q)`: cross-entropy = irreducible entropy + excess bits.
- **KL divergence:** `KL(p‖q) = Σ p log(p/q)` — the extra bits from using q instead of
  the best code p. Asymmetric; `KL(p‖q) = ∞` when q assigns zero mass where p doesn't
  (one direction only). This asymmetry reappears as mode-covering (forward KL) vs
  mode-seeking (reverse KL) behavior in VAEs (Phase 10) and RLHF reward modeling
  (Phase 9).
- **Perplexity (NLP):** `PP = exp(H)` on an LM test set — the geometric-mean number of
  equally-likely next-token candidates the model effectively faces. PP of 20 ≈ choosing
  among 20 options each step. Only comparable across models with the *same tokenizer*
  (foreshadowing Phase 3).
- **MLE view of losses:** minimizing CE over token labels = maximizing `Σ log p_θ(x_i)`
  = MLE. Training a language model "to predict the next token" and "to assign maximum
  likelihood to the corpus" are the same objective stated twice. MSE training is MLE
  under a fixed-variance Gaussian likelihood — the "fixed variance" assumption is the
  quiet limitation behind why MSE is the wrong loss for classification.

## 4. Convexity, and why non-convexity is survivable

- Convex function: line segment between any two points on the graph lies above the graph.
  Then every local minimum is global — and deep nets are *not* convex, in parameter
  permutation and saddle-rich ways.
- **High-dimensional saddles:** in n dimensions, a critical point is a local minimum only
  if all n Hessian eigenvalues are positive. Random perturbation makes most critical
  points saddles with *some* negative directions — saddles, not minima, are the typical
  obstacle. SGD's noise helps escape them (relevant in Phase 1).
- **Symmetry:** `W` and `W·P` (permutation of hidden units) represent the same function —
  loss surfaces have `n!`-fold equivalent basins. "The" loss landscape is really an
  equivalence class of landscapes. This is why sharp-vs-flat-minimum arguments are subtle.

## 5. Notation used throughout this repo

| Symbol | Meaning |
|--------|---------|
| `x, h, c` | input, hidden state, cell state (batch omitted) |
| `W_xh, W_hh` | input→hidden, hidden→hidden weight matrices |
| `σ, tanh` | sigmoid, hyperbolic tangent |
| `⊙` | elementwise (Hadamard) product |
| `q, k, v` | query, key, value vectors |
| `Q, K, V` | stacked queries/keys/values (seq_len × d) |
| `d_k, d_v, h` | key dim, value dim, number of attention heads |
| `PE(pos, 2i)` | sinusoidal positional encoding entry |
| `β₁, β₂, ε` | Adam moment decay rates, numerical epsilon |

## Cold-quiz (Phase 0)

Answer out loud, no notes:
1. Derive `∂CE/∂z = ŷ − y` two ways (full Jacobian; log-sum-exp shortcut).
2. In one sentence each: entropy, cross-entropy, KL — then show CE = H + KL.
3. Why does one reverse sweep give every gradient, and what does it cost vs the forward pass?
4. What does the spectral norm of a weight matrix tell you that the Frobenius norm doesn't?
5. Perplexity e³ on a test set — interpret it, and state one condition required to
   compare it against another model's perplexity.
6. Weight decay: write its update, then preview exactly where Adam breaks the
   L2 ↔ weight-decay equivalence (you'll close this loop in Phase 1).
