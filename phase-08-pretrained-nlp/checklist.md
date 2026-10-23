# Phase 8 — Modern NLP & Pretrained Language Models (3 weeks)

Goal: the pretraining revolution — how one Transformer objective becomes a
general-purpose NLP system. You should understand what BERT's masking *buys*
(bidirectionality), what GPT's causal objective *buys* (generality + in-context
learning), and why the field reorganized around these two shapes.

## Concepts

- **The paradigm shift**: feature engineering → model engineering → **transfer
  learning on text**. Before: one model per task, from scratch, on small labeled
  sets. After: one big pretrained model, adapted per task. Why this was even
  *possible* in NLP when vision had it earlier (word embeddings were the first
  hint; ELMo was the bridge; ULMFiT made the recipe explicit; BERT made it standard)
- **BERT** (encoder-only, bidirectional):
  - Masked LM: predict 15% of tokens; the 80/10/10 rule (mask/keep/random — why
    each third exists: [MASK] never appears at inference → mismatch; random
    replacement forces genuine use of context)
  - Why bidirectionality matters: "the bank of the river" needs right context;
    causal models must see it one-sided. Encoder = deep bidirectional fusion.
  - Next-sentence prediction (and why it was later *removed* — RoBERTa's finding:
      it didn't help; the task was too easy / poorly constructed)
  - The [CLS] token + classification head; token-level vs sequence-level outputs
  - What BERT is *not*: a generator (no autoregressive head) — the shape constrains
    the task family (Phase 7 Part D, paid off)
- **GPT lineage** (decoder-only, causal):
  - GPT-1: pretrain + fine-tune (the LM objective as universal pretraining)
  - GPT-2: "language models are zero-shot learners" — the scaling hypothesis
    emerging; the zero-shot vs fine-tuned framing; control/prompts appear
  - GPT-3: in-context learning — no gradient updates, just demonstration examples
    in the prompt. The conceptual earthquake: a *frozen* model adapts via
    conditioning. (Connect to Phase 7's induction heads: in-context learning has a
    proposed circuit-level mechanism.)
  - Instruction tuning and the models that follow (GPT-3.5/InstructGPT → Phase 9)
- **T5** (encoder–decoder, the systematic study): the text-to-text framework —
  every task as an input-prefix → output-string; span corruption vs causal LM
  objectives compared *in one paper*; the comprehensive recipe-comparison value
  of the paper
- **The fine-tuning mechanics** (conceptual, matching your no-code constraint):
  - The full fine-tuning recipe: replace/add task head, train all layers, small LR
  - Head-only vs full fine-tuning: the spectrum; catastrophic forgetting in NLP
    fine-tuning (and why it's milder than vision — the "fine-tuning is stable"
    result from ULMFiT/GPT-3 era); the frozen-encoder + linear-probe baseline
    everyone should compare against
  - The task zoo mapped onto shapes: token classification (NER, POS) = token
    outputs + CRF consideration; sentence classification (SST, MNLI) = [CLS] or
    mean-pool + head; span QA (SQuAD) = start/end logits over tokens; seq2seq
    (translation, summarization) = encoder–decoder models
  - Why QA format is a *language modeling* problem in GPT-style models too
    (span extraction reframed as generation)
- **Evaluation, honestly**: GLUE/SuperGLUE and their lessons (saturation, then
  adversarial/superGLUE extensions); the dev-set-overfitting treadmill;
  contamination (was the benchmark in pretraining data?); the move to human
  eval and LLM-as-judge; intrinsic (perplexity) vs extrinsic (task) metrics and
  their frequent disagreement
- The scaling foreshadow: what happens when fine-tuning is too expensive →
  prompting → LoRA → full RLHF pipeline (all Phase 9)
- Multilingual: cross-lingual transfer via shared vocab (XLM-R), the
  tokenizer-fairness issue (Phase 3's tokenization injustice at scale)

## Derivations / constructions (on paper)

- [ ] BERT's 80/10/10 rule: for a [MASK]ed token, write the training-time input
      and the target; then explain, case by case, what each of the three
      replacements contributes (do this as a table)
- [ ] The fine-tuning gradient story: why a small LR and warmup are standard for
      fine-tuning (pretrained weights are a good basin; large LR destroys it —
      connect to Phase 1's loss-landscape discussion)
- [ ] T5's text-to-text formatting for 4 tasks (classification, regression, QA,
      summarization) — write the exact input/output strings; then say what the
      unified format buys (one model, one objective, multi-task mixtures)
- [ ] GLUE's score construction: which tasks, how they're averaged, and why a
      single-number leaderboard distorts (per-task variance; the mnli-m/mnli-mm
      distinction)

## Misconceptions to kill

- [ ] "BERT understands language bidirectionally, GPT doesn't" — both use the same
      block; the *objective's* information flow differs. GPT's representations
      are causal-per-position but still contextual. Precision matters here.
- [ ] "In-context learning is few-shot fine-tuning" — no gradients, no parameter
      change; the mechanism (induction-head-style retrieval over the prompt) is
      different from gradient descent. This distinction is load-bearing for
      Phase 9's RLHF story.
- [ ] "Fine-tuning overwrites the pretrained knowledge" — mostly it *adapts* it;
      the stability result is real but has failure modes (domain shift, small data
      with high LR). The catastrophic-forgiving behavior of text fine-tuning
      is itself an empirical puzzle.
- [ ] "GLUE score = language understanding" — contamination, annotation artifacts
      (the famous "negation words alone solve it" findings), and saturation all
      argue otherwise. Know at least one concrete annotation-artifact example.
- [ ] "More pretraining data automatically helps" — dedup and quality filtering
      matter enormously (Phase 9's data section connects; C4's documented issues).

## Thought experiments

- [ ] You must build a system that classifies support tickets AND summarizes them.
      With (a) 2015 tech, (b) BERT-era, (c) GPT-era — describe each system's
      architecture, data needs, and failure modes.
- [ ] BERT sees "The cat sat on the mat because it was tired" — show exactly which
      tokens inform the representation of "it" in BERT vs in a causal LM. What
      can each *not* do?
- [ ] A colleague claims their GPT-style model matches BERT on NER without any
      fine-tuning (prompted, in-context). What information does the prompt
      provide that a random-init model would lack? Where does that information
      physically live? (Phase 7's residual-stream frame.)
- [ ] Your fine-tuned model scores 94% on the dev set and 71% on a freshly
      collected test set from the same "distribution." Generate three hypotheses,
      ordered by likelihood, and the diagnostic for each.

## Cold quiz
1. BERT's MLM: the objective, the 80/10/10 rule with each third's purpose, and
      why bidirectionality is impossible for a causal LM.
2. The GPT-1→2→3 conceptual progression in three sentences each, ending with
      in-context learning — then distinguish it mechanistically from fine-tuning.
3. T5's text-to-text idea: format two different tasks, and one benefit of the
      unified format.
4. The fine-tuning recipe: LR/warmup choices and the loss-landscape reason
      (Phase 1 connection).
5. The task zoo: model shape for NER, sentence classification, span QA, translation
      — encoder-only or seq2seq, and the output layer each uses.
6. Three reasons a GLUE number can mislead, with one concrete mechanism each.
