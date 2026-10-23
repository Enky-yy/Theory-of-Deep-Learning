# Phase 6 — Seq2Seq, Encoder–Decoder & the Attention Patch (1.5 weeks)

Goal: the architecture that *created the need* for attention. You should understand
the context-vector bottleneck precisely, be able to explain Bahdanau attention as an
engineering fix to that bottleneck, and see why the fix — generalized — becomes the
Transformer. This phase is the hinge of the whole course.

## Concepts

- **The encoder–decoder abstraction**: encoder maps a variable-length input sequence
  to a representation; decoder maps that representation to a variable-length output
  sequence. Decoupled lengths; shared interface. (Translate, summarize, dialogue,
  speech-to-text, image captioning — same skeleton, different modalities.)
- The RNN version: encoder reads x_1..x_n into final hidden state; that single vector
  *is* the representation; decoder initialized from it, emits y_1..y_m autoregressively
- **The context-vector bottleneck — the central failure**: a whole sentence compressed
  into one fixed vector (e.g., 256 floats). Long inputs: information must be
  *selectively discarded*; performance plateaus with length (Sutskever et al. 2014
  worked to ~20–30 tokens; the degradation curve is measurable)
- Teacher forcing at training vs free-running inference; exposure bias revisited
  (now with the BLEU impact)
- **Attention, Bahdanau-style**: at each decoder step, compute compatibility scores
  between current decoder state (query) and *all* encoder hidden states (keys);
  softmax → weights α_t,s; context vector c_t = Σ_s α_t,s·h_s (a *weighted average*
  of encoder states). Decoder input becomes [embedding; c_t]. The decoder now looks
  back at the *whole* input, differently at every step. The bottleneck is bypassed,
  not removed.
- Additive (Bahdanau: MLP score) vs multiplicative (Luong: dot/general/bilinear
  scores) compatibility — the direct ancestors of dot-product attention in Phase 7
- What attention *bought*: length-independent information access, aligned
  interpretability (the heatmap *is* the alignment learned — often strikingly
  monotonic for translation), better long-sentence BLEU
- What attention *cost*: O(n·m) per step (linear in sequence length then; quadratic
  in self-attention later — the seed of Phase 7's scaling problem)
- **Beam search**: why greedy decoding is locally optimal only; beam width k as
  approximate search; length normalization (unnormalized scores favor short outputs —
  probabilities multiply, so log-probs sum negative; derive the bias); the practical
  k=4–10 range; why beam search *hurts* BLEU in some tasks (Li et al. 2016: beam
  search prefers generic, high-probability outputs — a diversity problem, not a
  search problem)
- Exposure bias and its proposed cures: scheduled sampling (and its theoretical
  flaws), minimum risk training — know the names, understand the mechanism, know
  that none fully won
- Seq2seq evaluation: BLEU (modified n-gram precision + brevity penalty; what it
  measures and what it misses — no semantics, no synonymy), chrF, and the modern
  learned-metric turn (BLEURT, BARTScore) previewing Phase 8's evaluation crisis

## Derivations (on paper)

- [ ] The attention computation as four lines of math (scores → softmax → weights →
      context), then as shapes: h_s ∈ R^{n×2d}, query ∈ R^{2d}, α ∈ R^{n}, c ∈ R^{2d}
- [ ] Why unnormalized log-probability scoring biases beam search toward short
      sequences: two candidate outputs, lengths 3 and 10, average log-prob −0.2 each;
      compare totals; then fix it with length normalization and discuss the α choice
- [ ] BLEU by hand: a candidate sentence, 4 references, unigram precision, the
      clipping rule, brevity penalty — one full numeric example
- [ ] Count the attention cost at each decoder step: O(n·d) score + O(n·d) weighted
      sum; total for m output steps. Contrast with the constant-cost (but lossy)
      no-attention baseline

## Misconceptions to kill

- [ ] "Attention makes the model look at the input like a human" — it's a learned
      soft weighted average, differentiable, trained end-to-end; the heatmaps are
      *suggestive*, and probing studies show the weights don't always reflect causal
      dependence (faithfulness debates — know this before Phase 7's interpretability
      work).
- [ ] "The context vector is a new idea in attention" — it existed in the
      no-attention baseline (as the final state). What's new is a *different
      context vector at every decoder step*, computed from all encoder states.
- [ ] "Attention solved the vanishing gradient problem" — it solved the *information
      bottleneck*; gradients from decoder to early encoder steps get a direct
      path too, but the primary framing here is information, not gradients.
- [ ] "Beam search is always better than greedy" — see the diversity failure;
      and in the LLM era, sampling (temperature, top-p) displaced beam search
      entirely for open-ended generation (Phase 9).

## Thought experiments

- [ ] Design an input-output pair where attention must be *non-monotonic* (e.g.,
      reversal task, or a language with verb-final structure). What would the
      attention heatmap look like, and what would a monotonic-attention model do?
- [ ] The encoder output is a single vector and attention is removed. At what input
      length does translation quality visibly drop, and what *kind* of information
      is lost first (function words? long-distance agreement? name bindings?).
      Argue from the bottleneck math.
- [ ] Two beams in search: candidate A (short, high-probability, generic:
      "I don't know") vs candidate B (longer, lower per-token probability, specific).
      Which does unnormalized scoring pick? Which *should* be picked? What does
      this predict about LLM degeneration, and why did Phase 9's models switch
      to sampling?
- [ ] You're explaining attention heatmaps to a skeptic who says "those are just
      alignment diagrams from the 1980s, renamed." What's *computationaly* new?

## Cold quiz
1. The context-vector bottleneck: the mechanism, the length-dependence, and the
      number ~20–30 tokens where 2014-era seq2seq visibly struggled.
2. Write Bahdanau attention in four lines of math, with shapes, then name what's
      new compared to the no-attention baseline (two things: per-step context; all
      states, not just the final one).
3. Additive vs multiplicative compatibility: the formulas, and the descendant in
      the Transformer.
4. Why does unnormalized scoring bias beam search short? Give the fix and its
      hyperparameter.
5. BLEU: its four ingredients, one thing it measures well, two things it misses.
6. Name the concrete 2016 finding about beam search and translation quality, and
      its diagnosis (diversity vs search).
