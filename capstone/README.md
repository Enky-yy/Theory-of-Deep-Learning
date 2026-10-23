# Capstone (2–4 weeks)

The final test: **teach it**. Choose one option (or combine), and produce a written
artifact at a standard you'd be willing to publish. Nothing here is code — these are
written, derivation, and analysis deliverables that prove the curriculum stuck.

## Option A — The complete architecture essay
Write ~8,000–12,000 words rebuilding the modern LLM stack from first principles for
a competent engineer who hasn't seen this material:
- Math floor → optimizer → CNN (inductive bias interlude) → NLP foundations →
  RNN/LSTM → seq2seq → attention → Transformer → pretraining → alignment →
  inference → RAG
- Every major formula re-derived in the text (not cited — derived), every design
  decision traced to the failure it fixed
- The "why not X" counterfactual at every step (why not RNNs at scale? why not
  beam search in LLMs? why not full fine-tuning for every adaptation?)
- Rubric: a reader should be able to *pass your Phase 7 cold quiz* from your essay alone

## Option B — The deep comparison paper
Pick one contested question from the course and write a rigorous position paper
(~5,000–8,000 words) defending a side, with both sides steelmanned:
Candidates (or bring your own):
- AdamW vs SGD+momentum at scale: is the optimization gap real or tuning-artifact?
- Decoder-only supremacy: technical inevitability or path dependence?
- The emergence debate: real phase transitions or metric artifacts?
- RAG vs long context: where does each die?
- Attention-heatmap interpretability: valid evidence or seductive decoration?
- RLHF vs DPO: is the RL machinery earning its complexity?
- The quadratic attention wall: will state-space models displace Transformers,
  or is the wall irrelevant at practical context lengths?

## Option C — Paper reproduction, on paper
Choose one primary paper from RESOURCES.md you have *not* fully understood, and
produce:
1. A complete mathematical reconstruction (every theorem/derivation, gaps filled)
2. The "what problem, what was known, what this changed" historical frame
3. A critique section: the assumptions that don't hold, the experiments that
   don't prove the claim, what the follow-up literature found
4. An appendix connecting it to at least two later course topics it enabled

## Shared requirements (all options)
- [ ] Every cold-quiz question from all phases that touches your chosen topic is
      answerable from your artifact's content — test yourself cold first
- [ ] At least three "I used to think X, now I think Y" corrections, with the
      exact source (paper, derivation, or thought experiment) that flipped each
- [ ] One full-page summary diagram (drawn by hand, photographed is fine) of your
      topic's mechanism — no copied figures
- [ ] A final self-audit: list the 5 weakest entries in your entire curriculum
      checklist and what you'd do about them

## Completion
When the artifact is done, update [PROGRESS.md](../PROGRESS.md), then run the
course-final quiz: 20 random cold-quiz questions from every phase, answered out
loud, in one sitting. That's the finish line.
