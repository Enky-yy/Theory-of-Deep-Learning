# Phase 9 — LLMs: Pretraining to Deployment (3 weeks)

Goal: the full modern LLM stack, conceptually — data → scaling → instruction
tuning → preference alignment → parameter-efficient adaptation → efficient
inference → retrieval systems. Every pipeline stage's *reason*, its *failure
modes*, and its place in the 2020s lineage.

## Concepts

### 9a. Pretraining at scale
- The data pipeline: Common Crawl → filtering (quality classifiers, language ID,
  dedup — exact AND fuzzy/MinHash; why dedup is worth percentage points of loss),
  toxic-content filtering (and its documented over-filtering of marginalized
  dialects), data-mixture ratios (code in a text model? why — reasoning and
  formatting benefits), the "data wall" discussion (finite high-quality text,
  synthetic-data loops, licensing)
- Tokenization at scale revisited: BPE/SentencePiece choices, vocab size
  trade-offs (50k–256k), the embedding-parameter share of the model, why
  modern models shrink embedding tables (tied embeddings, LoRA on embeddings)
- **Scaling laws, the real content**:
  - Kaplan et al.: loss as a power law in N, D, C — smooth, predictable over
    *many* orders of magnitude; compute-optimal vs parameter-optimal confusion
  - **Chinchilla (Hoffmann)**: the correction — most 2020-era LLMs were
    *over-parameterized and under-trained*; optimal ≈ 20 tokens per parameter;
    the D = 2N rule and what it changed (Chinchilla 70B > Gopher 280B);
    practical implications (inference-cheaper models train longer)
  - The emergent-capability debate (Wei et al. vs the critique that "emergence"
    is a metric artifact — smooth under continuous metrics): know both sides
- The training run itself: the loss-curve discipline (predictable power-law
  decline; a bump = data or numerics bug), mixed-precision (bf16) at scale,
  parallelism taxonomy (data/tensor/pipeline parallelism — what each splits and
  what each costs in communication), checkpointing and fault tolerance
  (thousands of GPUs die mid-run; the run is a distributed-systems artifact)

### 9b. Alignment: instruction tuning → RLHF → DPO
- Why pretraining alone isn't a product: the model *completes* text, it doesn't
  *follow instructions*; base models answer questions with more questions
- **Instruction tuning**: SFT datasets (supervised fine-tuning on prompt→ideal-
  response pairs); quality > quantity (the LIMA finding — 1k carefully chosen
  examples competitive with 50k+); format and refusal patterns learned here
- **RLHF in full** (Ouyang et al. pipeline):
  1. SFT (base → instruction-following)
  2. Reward model: human preference pairs (A vs B responses to a prompt) →
     Bradley-Terry model → RM scores responses; the economics of human labeling;
     RM overoptimization (Goodhart: optimizing the proxy beyond its validity
     region — the KL penalty to the reference policy is the guardrail; state
     why exactly)
  3. PPO fine-tuning: maximize RM score under a KL constraint to the SFT policy;
     the PPO-specific pieces (clipping, value head, GAE) at a conceptual level;
     the reward-hacking pathologies (length exploitation, sycophancy, format
     gaming) — every one is a documented phenomenon with a mechanism
- **DPO**: the derivation-level insight — the optimal policy under the
  Bradley-Terry objective has a closed form relating policy ratio to reward;
  substituting it yields a *classification-style* loss on preference pairs
  (preference likelihood), no RM, no RL loop. The stability/consequence
  trade-off vs RLHF (simpler, but no online exploration; can still overfit
  preferences).
- Constitutional AI / RLAIF: AI feedback replacing human feedback; the
  scalable-feedback argument and its circularity risk
- What alignment does to capability: the "alignment tax," jailbreaks as
  adversarial examples against the alignment layer, and the honest framing:
  alignment is a surface behavior layer over the base model, removable via
  fine-tuning (the base/aligned duality)

### 9c. Parameter-efficient fine-tuning & quantization
- **LoRA**: the low-rank hypothesis (weight *updates* during adaptation are
  low-rank — empirically rank 4–16 suffices); W' = W + BA with B: d×r, A: r×d;
  the parameter arithmetic (r(d_in+d_out) vs d_in·d_out — get the ratio for
  GPT-3 12288, r=8); why the A-zero/B-zero initialization matters (start as
  exact identity); where LoRA is applied (attention projections, then FFN;
  embedding & LM-head LoRA); the merge-at-inference property (zero added
  latency); QLoRA (quantized base + LoRA adapters; double quantization;
  NF4)
- The broader PEFT picture: adapters, prefix/prompt tuning — where each lost
- **Quantization**: why weights fit in fewer bits (outlier structure, the
  quantization-error argument); GPTQ-style post-training quantization (layer-wise
  error minimization, the observer/hessian idea conceptually); the 4-bit
  inference ecosystem; what quantization *costs* (long-context, some tasks
  degrade more)

### 9d. Inference & serving
- **KV caching**: the autoregressive recomputation waste (each new token
  recomputes attention over the whole prefix — O(n²) total work per token
  generation); the cache = every layer's K and V for every past token;
  the memory arithmetic that makes it the *serving* bottleneck (per-token
  per-layer KV bytes × layers × batch — do this calculation);
  prefill vs decode phases (compute-bound vs memory-bound; why decode is
  memory-bandwidth-bound and what that implies for hardware choices)
- **PagedAttention/vLLM**: the KV cache as an OS-paging problem (fragmentation,
  preallocation waste); continuous batching
- **Speculative decoding**: a small draft model proposes k tokens, the big model
  *verifies* them in one forward pass via exact-acceptance sampling (the
  mathematical guarantee: output distribution *unchanged*; derive the acceptance
  probability); when it helps (parallel verification vs sequential draft cost)
- **Sampling as decoding**: temperature (scaling logits — the softmax geometry
  from Phase 7, again), top-k, **nucleus/top-p** (the cumulative-probability
  truncation; why it replaced beam search for open-ended text — connect to
  Phase 6's beam-search diversity failure); repetition/frequency penalties and
  why they're hacks
- **Long context**: RoPE interpolation/scaling (NTK-aware scaling — the rotation
  machinery of Phase 7, made practical), the quadratic-cost return (FlashAttention
  helps constants not the asymptotic), attention sinks, position-dependent recall
  ("lost in the middle"), the RAG-vs-long-context trade-off (Phase 9e)

### 9e. RAG, tool use, agents (system design)
- The retrieval pipeline: embedding model (contrastive-trained sentence
  encoders — Phase 3's distributional hypothesis at scale), chunking strategy
  (fixed vs semantic; the granularity trade-off), vector index (ANN/HNSW at
  concept level), top-k + reranking (cross-encoder rerankers: why two-stage
  retrieval), grounding the generation and prompt design for citation
- When RAG vs long-context vs fine-tuning: the decision matrix (fresh/private
  knowledge → RAG; style/behavior → tuning; stable long docs → context)
- RAG failure modes: retrieval misses → hallucination anyway (the model doesn't
  know what it doesn't know); distractor documents; the evaluation problem
  (faithfulness vs relevance vs answer correctness — three different metrics)
- Agents & tool use: the loop (model emits action → environment returns
  observation → repeat); function calling as constrained generation; ReAct-style
  reasoning-then-acting; the reliability cliff (error compounding over steps —
  the Phase 6 exposure-bias intuition, now at the system level)

## Derivations / calculations (on paper)

- [ ] Kaplan vs Chinchilla: the compute-optimal relation C ≈ 6ND, then solve:
      fixed budget 10²³ FLOPs — parameter count and token count for the optimum;
      then contrast with "just make it bigger" (GPT-3's D/N = 20 vs Chinchilla's 20
      tokens/param — wait, compute the actual ratios and state GPT-3's deficit)
- [ ] KV cache memory: d=4096, L=32 layers, 32 heads, fp16, context 4k, batch 32 —
      total KV cache bytes for one generation. (This number explains the entire
      serving-infra industry.)
- [ ] LoRA parameter ratio: dense 12288×12288 vs LoRA r=8 — exact parameter counts,
      ratio; then why A=0/B=0 init preserves the base model exactly
- [ ] Speculative decoding acceptance probability: derive that with acceptance
      α = min(1, p/q) per token the output distribution equals p's (the standard
      speculative-sampling argument)
- [ ] Bradley-Terry → DPO: from p*(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l)) and the
      optimal-policy closed form π* ∝ π_ref·exp(r/β), eliminate r → the DPO loss
      (the key algebraic move: the log-ratio *is* the implicit reward)
- [ ] Temperature's effect: logits [2, 1, 0.5] at T=1 vs T=0.5 vs T=2 — softmax
      by hand; connect to Phase 7's scaling discussion (temperature is the same
      knob, user-controlled)

## Misconceptions to kill

- [ ] "Scaling laws predict capability" — they predict *loss*. The loss→capability
      mapping is the contested step (smooth loss, sometimes discontinuous
      capability — the emergence debate). Keep the two statements separate.
- [ ] "RLHF makes models smarter" — it changes *behavior distribution*, not
      knowledge; the base model's capability is what it is; RLHF trades a little
      capability (alignment tax) for usability, and can induce sycophancy by
      optimizing for human-pleasing proxies.
- [ ] "DPO is just better RLHF" — different trade-space: simpler and stable, but
      constrained to the preference data's support; RLHF with a reward model can
      generalize beyond fixed pairs (and overoptimize them differently).
- [ ] "LoRA works because low-rank is a fundamental truth" — it works because
      *task adaptation* moves weights little and in few directions; full-rank
      changes are needed for distant domains; rank is a tunable capacity knob,
      not a law.
- [ ] "RAG solves hallucination" — it shifts the failure from *knowledge* to
      *retrieval-and-faithfulness*; models still ignore or misattribute provided
      context; and hallucination *rates* drop, not to zero.
- [ ] "Quantization is free" — average loss is tiny but outlier channels matter;
      specific tasks (long context, math) degrade more; always re-evaluate after
      quantizing.
- [ ] "In-context learning = the model training during generation" — no parameter
      update; the mechanism is retrieval/conditioning over the context (Phase 7's
      induction heads, Phase 8's distinction). The word "learning" misleads.

## Thought experiments

- [ ] You have 10²³ FLOPs. Chinchilla-optimal says a mid-size model, long training.
      A competitor spends the same compute on a 10× larger model, 10× less data.
      Predict the benchmark profile (which tasks each wins) and the serving cost
      difference.
- [ ] An RLHF-trained assistant keeps agreeing with users even when wrong. Trace
      the mechanism through the pipeline: which stage produced it (RM data? RM
      generalization? PPO's KL budget?), and which intervention fixes it (data,
      reward hacking penalty, DPO with contrastive pairs?).
- [ ] Your RAG system answers correctly when retrieval works, and *confabulates
      citations* when it doesn't. Design the system change that makes abstention
      likely, and explain why prompting alone is weak here.
- [ ] Speculative decoding with a draft model that's 3× smaller but accepts only
      40% of its proposals: is it worth it? Do the arithmetic (draft cost vs
      saved big-model steps) and find the break-even acceptance rate.

## Cold quiz
1. Chinchilla in one precise statement + what it said about 2020-era models; then
      why the loss→capability leap needs care (emergence debate, both sides).
2. The full RLHF pipeline: three stages, the RM's mathematical form (Bradley-Terry),
      the KL constraint's exact purpose (Goodhart guardrail).
3. DPO: the core algebraic move (log-ratio as implicit reward) and its practical
      consequence (no RM, no RL loop); one trade-off vs RLHF.
4. KV cache: what's cached, why decode is memory-bound, and the number that makes
      it the serving bottleneck (do the arithmetic from memory's order of magnitude).
5. LoRA: the rank hypothesis, the init trick, the merge property; parameter
      ratio for r=8 vs dense at d=12288.
6. Sampling: temperature, top-k, top-p — mechanism of each, and why beam search
      lost to sampling for open-ended text (Phase 6 connection).
7. RAG vs long-context vs fine-tuning: the one-line decision rule for each.
