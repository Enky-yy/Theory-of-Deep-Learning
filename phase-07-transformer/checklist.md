# Phase 7 — The Transformer: Self-Attention Deep Dive (4 weeks — the centerpiece)

Goal: every mechanism in the Transformer explained from its equation and its *reason*.
By the end you should be able to rebuild the architecture from memory, justify every
design decision against the failure it fixed, and handle the positional-encoding
math completely. This phase is weighted heavier than any other.

---

## Part A — Scaled dot-product attention (week 1)

### Concepts
- **Q, K, V from first principles**: attention as *soft dictionary retrieval*.
  Query = "what am I looking for now"; keys = "what do I contain"; values = "what
  do I hand over if selected". The retrieval analogy, made exact: scores = q·k,
  weights = softmax(scores), output = weighted sum of v.
- The three roles of softmax here: normalization (weights sum to 1 = convex
  combination), sharpening (relative scores → nonlinear weights), differentiability
  (no argmax — the whole thing trains end-to-end)
- **Self-attention**: Q, K, V all come from the same sequence. Each position asks
  "given everything (including me), what should I attend to?" — one step of
  *global* mixing vs CNN's local, RNN's sequential. The path-length argument:
  any two positions interact in **one** layer — compare RNN's O(n) and CNN's
  O(log_k n) with kernel k. This single fact is why transformers parallelize
  (no sequential recurrence) and why they scale.
- **The √d_k scaling — derive it properly**: qᵀk is a sum of d_k terms; if
  q_i, k_i ~ independent, mean 0, var 1, then Var(qᵀk) = d_k → magnitudes grow
  with dimension → softmax saturates into near-one-hot (gradient → 0). Divide by
  √d_k restores Var = 1 regardless of dimension. The 2017 paper's "sufficiently
  large" is exactly this. Then: why saturating softmax is bad (gradient flow),
  and the connection back to Phase 0's softmax geometry.
- Causal masking in self-attention: setting pre-softmax scores to −∞ (not 0 —
  why not 0?) above the diagonal; why the decoder needs it (autoregressive LM,
  no peeking at future tokens) and the encoder doesn't
- Padding masking: same −∞ trick for [PAD]; why masking *before* softmax, never after
- Full attention-as-matrix form: softmax(QKᵀ/√d_k)·V — one expression, all the
  shapes: Q, K ∈ R^{n×d_k}, scores ∈ R^{n×n}, V ∈ R^{n×d_v}, out ∈ R^{n×d_v}.
  The n×n weight matrix *is* the "attention map" you'll visualize later.
- Computational cost: O(n²·d) time, O(n²) memory for the score matrix — the
  quadratic wall, stated now, revisited in Part E (and Phase 9's efficiency work)

### Derivations (on paper)
- [ ] Var(qᵀk) = d_k under the i.i.d. assumption — full derivation, then the √d_k fix
- [ ] Softmax saturation: compute softmax([1,2,3]) vs softmax(10·[1,2,3]) vs
      softmax([1,2,3]/√d) for d = 64 — numerically, and sketch the gradient
- [ ] The full attention computation with concrete shapes: n=8, d_k=64, d_v=64 —
      write every intermediate tensor's shape
- [ ] Why −∞ masking (not 0): softmax(e^x) — what weight does a masked entry get
      with score 0 vs score −∞? Do the arithmetic.

### Misconceptions to kill
- [ ] "Attention weights are probabilities over importance" — they're normalized
      *compatibility* scores, computed from query-key dot products; "importance"
      is an interpretation, and probing shows weights ≠ causal influence
- [ ] "Self-attention replaces RNNs because it's faster" — it's *parallelizable*
      (O(1) sequential steps vs O(n)); total FLOPs can be higher. Different axis.
- [ ] "√d_k is just a fudge factor that helps" — it's specifically variance
      normalization; the derivation is four lines and you should own it.

### Cold quiz (Part A)
1. Q/K/V: the retrieval analogy, stated precisely, then the equation, then the shapes.
2. Derive Var(qᵀk) = d_k and the √d_k fix; say what breaks without it (two things:
      saturation, gradient).
3. Path length for information to travel between positions: RNN vs CNN vs
   self-attention, with the formulas.
4. Causal mask: why −∞, and why before softmax?
5. State the n² quadratic cost for both time and memory, and name what the n²
   matrix *is* (the attention map).

---

## Part B — Multi-head attention & positional information (week 2)

### Concepts
- **Multi-head attention**: split d_model into h heads of d_k = d_model/h; run
  h independent attentions in parallel; concatenate; project back. Why: one
  attention per head is a *single* convex combination of value vectors — a rank
  limitation; h heads give h simultaneous, independent retrieval patterns.
  (Precise form: heads can attend to *different positions* — one head tracks
  syntax, another coreference; empirical head-specialization results — Clark et al.
  2019: some heads are positional, some syntactic, some rare-token; and the
  striking result that many heads are ablatable with little loss.)
- The h·d_k = d_model bookkeeping; the output projection's role (mixing heads);
  parameter count of MHA vs single-head
- **Positional information — the deepest point in the course**:
  - Why attention is permutation-invariant: the output is a weighted sum of value
    vectors, and softmax-weights follow scores; permute the input → permute rows
    of weights → permute output identically. *No order information survives.*
    (Prove it on paper — Part B derivation #1.) Recurrence had order built in;
    CNN had position via locality; attention throws both away and must be *told*.
  - **Sinusoidal absolute encodings** (the original): PE(pos, 2i) = sin(pos/10000^{2i/d}),
    PE(pos, 2i+1) = cos(...). Properties to own: each dimension is a sinusoid of a
    geometrically-spaced wavelength (from 2π to 10000·2π); position pos maps to a
    unique pattern; **relative offsets are linear transforms**: PE(pos+k) is a
    fixed linear function of PE(pos) — same rotation for every pos (each pair of
    dims (2i, 2i+1) is a 2D rotation by k·ω_i — this is exactly RoPE's idea made
    explicit). The paper's hypothesis: the model can *learn* to use relative
    offsets because linearity makes them easy to attend to.
    [1] The sin/cos pair-per-frequency structure: why sin and cos *both* — they're
    the (x, y) coordinates of a point on a circle; rotation matrix needs both.
    [2] The uniqueness/periodicity argument: frequencies geometrically spaced so no
    two positions collide across all dimensions.
    [3] The rotation interpretation: PE(pos+k) = R_k · PE(pos), same R_k for all pos.
  - Learned absolute embeddings (BERT, GPT-2 style): a lookup table max_len × d_model;
    simpler, empirically fine, but no extrapolation past max_len (the model has
    never seen position 1500), and no relative-structure prior.
  - **RoPE (rotary, 2021)**: instead of *adding* position to the embedding,
    *rotate* each q and k by position-dependent angles, so that the attention score
    q(pos)ᵀk(pos') depends only on (pos − pos') — a relative encoding that lives in
    the score, exact, no added parameters, and built from the sinusoidal identity.
    This is why essentially every modern LLM (LLaMA, Qwen, Mistral…) uses RoPE.
  - Relative alternatives: Shaw et al. (learned relative biases added to scores),
    T5 relative bias (learned per-distance buckets), **ALiBi** (fixed linear penalty
    on distance, no learned parameters, strong length extrapolation) — the design
    space and the trade-offs (extrapolation vs in-context learning quality).
  - Position for cross-attention (decoder's queries get decoder positions; encoder
    K/V already position-bearing — where does position enter the decoder?)
  - 2D positional encoding for images (ViT's learned pos emb; the RoPE-for-vision
    variants) — one line each.

### Derivations (on paper)
- [ ] Permutation invariance of self-attention: full proof (permute rows of K and V,
      track what happens to scores, softmax rows, and the output)
- [ ] Sinusoidal PE: derive PE(pos+k) = R_k·PE(pos) for one frequency pair — the
      rotation-matrix multiplication, from the angle-addition identities
- [ ] Why sin *and* cos: show that sin alone cannot give the linear relative
      property (construct the failure)
- [ ] Multi-head parameter count: d_model=512, h=8 → per-head dims, concat dim,
      output projection size; total MHA parameters vs single-head attention
- [ ] RoPE: derive that (R_p q)ᵀ(R_m k) depends only on (p − m), using the same
      rotation trick as above but applied to q and k separately

### Misconceptions to kill
- [ ] "Multi-head attention is just a wider attention" — the concatenation + output
      projection means heads compute *independent* weightings; width alone (one
      softmax over d) wouldn't give that. The softmax-per-head is the point.
- [ ] "Sinusoidal PE adds position to the token embedding" — true for the 2017
      original (additive); RoPE doesn't add anything, it *rotates* q/k inside every
      attention score. Two different mechanisms, both called "positional encoding."
- [ ] "Transformers can't extrapolate to longer sequences" — sinusoidal and learned
      absolute struggle; ALiBi extrapolates well; RoPE *with interpolation scaling*
      handles long context. The statement needs the encoding named.
- [ ] "Positional encoding is a minor detail" — without it, the model is provably
      blind to order (Part B derivation #1); for a language model that's fatal.

### Cold quiz (Part B)
1. Prove self-attention is permutation-invariant; then say in one sentence what
      positional encoding restores and *how* (added vs rotated — two mechanisms).
2. Multi-head: why independent softmaxes per head beat one big softmax? (Rank
      argument, stated concretely.)
3. Sinusoidal PE: write the formula from memory; derive the relative-offset
      rotation property; say why sin+cos (not sin alone).
4. RoPE vs sinusoidal-additive: the mechanism difference, and why RoPE wins for
      LLMs (relative scores *exactly*, no length table, extrapolates better).
5. ALiBi in one sentence, and what it's best at.
6. Where do positional embeddings enter a standard decoder's cross-attention?

---

## Part C — The full block & the encoder (week 3)

### Concepts
- The full Transformer block, every piece with its reason:
  - **Pre-norm vs post-norm** (original paper = post-norm: x + Sublayer(LayerNorm(x))?
    No — original is x + Sublayer(x), then LN; modern/pre-norm is
    x + Sublayer(LayerNorm(x))). Why pre-norm won: the residual path becomes a clean
    identity highway for gradients (the LN doesn't sit on the main highway); training
    stability without warmup crutches; the loss curves and the gradient-norm curves
    that show it.
  - **LayerNorm**: normalize per token across its d features: (x − μ)/√(σ²+ε)·γ + β;
      vs BatchNorm's per-channel across batch (Phase 2) — the *why* restated with
      the sequence-modeling reason: variable lengths, batch-of-1 inference, no batch
      statistics to maintain. Then the modern wrinkle: RMSNorm (LLaMA et al. — drop
      mean-centering and the bias, keep the scale; cheaper, empirically equal/better)
  - **The FFN block**: x → W₁ (d → 4d) → nonlinearity → W₂ (4d → d): a per-position
      two-layer MLP, *the same weights at every position* (like conv's sharing, but
      pointwise). The 4× expansion factor and its rationale. Modern variants:
      SwiGLU (gated FFN, used in nearly every LLM now — extra parameters, better),
      GeGLU; the "FFN as key-value memory" interpretation (Geva et al. 2021 — W₁
      writes patterns, W₂ reads them out)
  - Residual connections: the Phase-2 argument, now with the norm placement subtlety
  - What the *stack* does layer-by-layer (evidence from interpretability:
      lower layers = local/syntactic, middle = semantic, upper = task/formatting;
      logit-lens results)
  - Dropout placement: after attention weights, after FFN, on residuals — the 2017
      choices vs modern LLM practice (dropout ≈ 0 at scale; a real change worth noting)
- **The encoder**: N identical blocks (MHA + FFN, no causal mask) + final LN;
  representation = contextualized embedding per token. The BERT-shaped thing.
- Attention as a *differentiable information bus*: attention writes information from
  one position's residual stream into another's; the residual stream is the shared
  memory all sublayers read/write (Elhage et al.'s frame — very high leverage for
  everything in Phase 8's interpretability)
- The complete parameter accounting: per-block params (at d=512: MHA 4d², FFN 8d²
  → FFN is 2/3 of the block) — and the scaling-law connection (Phase 9: parameters
  ≈ 12·L·d²)

### Derivations (on paper)
- [ ] Write the full encoder block (pre-norm) from memory: every sublayer, every
      residual, exact order; then the parameter count for d=512, h=8, FFN=2048
- [ ] LayerNorm vs BatchNorm: one formula each, then the axis of normalization in
      each — and the inference-mode consequence (no running stats for LN)
- [ ] Gradient through the pre-norm residual path: show the identity highway
      (dL/dx picks up an uncontended +I term); contrast with post-norm's path
- [ ] The 12·L·d² parameter law: derive it (MHA 4d² + FFN 8d² = 12d² per block,
      ignore embeddings/norms); use it to check: GPT-3 175B, d=12288, L=96 →
      confirm ~1.8B per block × 96

### Misconceptions to kill
- [ ] "Attention is the whole Transformer" — at d=512 the FFN is 2/3 of the block's
      parameters and compute; ablations show FFN layers matter as much.
- [ ] "LayerNorm normalizes across the batch" — that's BatchNorm; LayerNorm's
      per-sample, per-token normalization is precisely why it fits sequences.
- [ ] "The residual connection adds information" — it adds *nothing*; it guarantees
      an identity path so every sublayer only learns a *residual correction*.
      The information argument is about gradients and optimization, not capacity.
- [ ] "Modern LLMs are the 2017 architecture unchanged" — no: pre-norm, RMSNorm,
      SwiGLU, RoPE, GELU-not-ReLU, dropout≈0. Know the diff list cold; it's the
      difference between understanding 2017 and understanding 2025.

### Cold quiz (Part C)
1. Write the full pre-norm encoder block from memory, every residual and norm,
      in order. Then the parameter count at d=512.
2. Pre-norm vs post-norm: the gradient-path difference, and why pre-norm won.
3. LayerNorm vs BatchNorm: normalization axes (formula-level), and the sequence-
      modeling reason LN dominates.
4. The FFN block: shapes, expansion factor, why the same weights at every position,
      and one modern gated variant.
5. Derive 12·L·d² and verify it on GPT-3's published config.
6. What does attention write into the residual stream? State the read/write frame.

---

## Part D — Decoder-only, encoder-only, encoder–decoder (week 3–4)

### Concepts
- The three architectures from the same parts:
  - **Encoder-only** (BERT): bidirectional attention everywhere; [CLS] for
    classification, token outputs for tagging/QA; masked-LM objective (Phase 8).
  - **Decoder-only** (GPT): causal mask everywhere; next-token prediction; the
    LM everyone converged on (why: single-stack simplicity, scaling cleanliness,
    in-context learning emerges, easier RLHF — one coherent story).
  - **Encoder–decoder** (T5, BART): full attention in encoder, causal +
    cross-attention in decoder; seq2seq objectives (span corruption); still used
    where output length ≠ input length (translation, summarization).
  - Prefix-LM as the hybrid (BERT-style on prefix, causal on suffix).
- Cross-attention mechanics: Q from decoder, K/V from encoder — where exactly it
  sits in the decoder block; the fact that it's the *only* block with two sources
- What causal masking buys for free: the factorization p(x) = Π p(x_t | x_{<t})
  is *exactly* the training objective — no special structure needed, just a mask.
  The single most important sentence for Phases 8–9.
- Scaling the architecture: compute-optimal intuition preview (L vs d trade-off —
  Chinchilla in Phase 9); attention O(n²) in context length, MLP O(n) — where
  the compute actually goes at long context

### Derivations (on paper)
- [ ] Write the decoder-only block: differences from the encoder block (mask only,
      two norms? one norm? — get the exact GPT-2 form right: pre-LN, one LN at the
      end); then the BERT block's form
- [ ] Show that causal-masked self-attention's training loss = negative
      log-likelihood = Σ −log p(x_t | x_{<t}) — the token-factorization identity
- [ ] Encoder–decoder: count the attention sublayers per decoder block (3: causal
      self, cross, FFN-attention?) — be exact: self-attn, cross-attn, FFN
- [ ] Compute split at n=4096, d=4096, L=32: attention matmuls O(n²d) vs FFN
      O(nd²) — which dominates? (Do the numbers; the answer flips at long n.)

### Misconceptions to kill
- [ ] "BERT and GPT are different models" — they're the same parts with different
      masks and objectives; the mask choice forces the objective choice, which
      forces the task family.
- [ ] "Decoder-only won because it's better at language" — it won partly for
      systems reasons: one stack, in-context learning at scale, RLHF-friendliness,
      no encoder-decoder state juggling. Understand the technical AND practical chain.
- [ ] "Cross-attention is just another attention" — it's the only sublayer where
      two different sequences interact; in decoder-only models it doesn't exist at
      all (conditioning happens *in the sequence itself* — how does an LLM see an
      image then? multimodal token insertion, Phase 10).

### Cold quiz (Part D)
1. The three architectures from memory: mask pattern, objective, canonical model,
      one task each is suited to.
2. Write the exact GPT-style decoder-only block (pre-norm positions, norms count).
3. Show the causal-mask loss = autoregressive NLL. This is *the* identity linking
      masking to language modeling — derive it.
4. Where does cross-attention live in the encoder–decoder, and what does
      decoder-only conditioning look like instead?
5. At n=4096, d=4096: which part of the block dominates compute? At n=1M?

---

## Part E — Synthesis & interpretation (week 4)

### Concepts
- Rebuild the whole architecture from scratch on a blank page, twice, 24 hours apart —
  every equation, every shape, every design rationale. This is the phase's checkpoint.
- **Attention-map interpretation**: what the n×n matrix shows (head-specific
  retrieval patterns); the empirical zoo: induction heads (the key-value lookup
  circuit: attend to the token after a previous occurrence of the current token —
  the mechanistic basis of in-context learning, Elhage et al.); previous-token
  heads; positional heads; attention sinks (the surprising role of attending to
  BOS/first token in modern LLMs — a stabilization artifact)
- **Ablation thinking**: what does zeroing one head/layer do? The surprising
  robustness (redundancy), and what that means for "the model learned X" claims
- The interpretability caution from Phase 6, matured: attention weights ≠ causal
  importance; use ablation/patching as the ground truth; when heatmaps mislead
- Efficiency, understood conceptually (Phase 9 will use these): FlashAttention's
  IO-awareness (the n² matrix is never *materialized* — tiling + online softmax;
      understand the online-softmax trick specifically), KV caching (Phase 9),
      grouped-query attention (multiple Q heads sharing K/V heads — memory,
      not compute, is the serving bottleneck), MQA/MLA; sparse attention patterns;
      linear-attention lineage (the full circle back to RNNs — state-space models
      as gated convolutions, Mamba: the field's answer to the quadratic wall)
- The architecture's honest limitations: context window is *not* memory (recall
      vs in-context learning debates); quadratic scaling; the induction-head
      story vs true reasoning (Phase 9's chains-of-thought fit here conceptually)

### Derivations (on paper)
- [ ] Full architecture, from blank page, twice: every equation, shape, mask,
      rationale. 24h apart. If the second attempt differs anywhere, that entry
      wasn't mastered.
- [ ] The induction head's algorithm, written as attention mechanics: given
      sequence "the cat sat on the mat … the cat", show which (query, key) pairs
      produce the induction pattern, and why causal masking makes it possible
      (can only look *back*)
- [ ] Online softmax: the key identity — softmax over a stream can be updated
      incrementally with a rescaling term; write the update, then say what
      FlashAttention does with it (and why memory, not FLOPs, is the win)
- [ ] GQA parameter/memory math: MHA with h=32 heads vs GQA with 8 KV-heads —
      the KV cache size per token, ratio

### Misconceptions to kill
- [ ] "The attention heatmap tells you what the model is doing" — it tells you
      where a specific *sublayer* reads from; the causal story needs ablations.
- [ ] "Long context = good memory" — retrieval within the window is empirically
      weak ("lost in the middle"); position-dependent recall is a real phenomenon.
- [ ] "FlashAttention is an approximate/faster-math trick" — it's *exact*; the
      speedup is IO-aware tiling + the online-softmax trick; FLOPs actually increase slightly.

### Cold quiz (Part E, and the phase-final full quiz)
1. Blank-page rebuild: full decoder-only Transformer, every equation and rationale,
      on paper, from memory.
2. Induction head: the mechanism in attention terms, and why it's the proposed
      basis of in-context learning.
3. Why attention weights ≠ causal importance — one sentence, plus the correct
      method (ablation/patching).
4. FlashAttention: what's never materialized, the online-softmax identity, and
      whether it changes the math (it doesn't).
5. GQA: what it saves (memory/compute — be exact) and why serving cares.
6. The 2017→modern diff list, in full: every substitution (norm, activation,
      positional, RMSNorm, dropout) and one line on each why.
