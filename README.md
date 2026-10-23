# Mastering Deep Learning: Fundamentals → LLMs

A ~28-week, 20+ hrs/week, **theory-first** curriculum for deep conceptual mastery of:
optimizers, CNNs, RNNs (hidden state), LSTM/GRU (cell state), encoder–decoder architectures,
self-attention, positional encodings, NLP foundations, pretrained language models, LLMs,
and generative/multimodal models.

**This is a no-code curriculum.** You already know DL at an intermediate level — the gap
this fills is depth: being able to derive, explain, and defend every mechanism from first
principles. Work happens on paper, in conversation, and by reading papers — not in notebooks.

## Ground rules (how mastery actually happens)

1. **Derive it, don't memorize it.** Every formula in this curriculum has a checklist entry
   requiring you to re-derive it on paper (LSTM gates, √d scaling, Adam bias correction...).
   If you can't derive it, you don't know it.
2. **Say the shapes out loud.** Most DL confusion is tensor-shape confusion. Every mechanism
   gets explained with concrete dimensions at least once.
3. **Kill the misconceptions.** Each phase lists the specific wrong beliefs people carry
   between phases. Confront each one explicitly.
4. **Cold-quiz checkpoint.** Before marking a phase done, answer its quiz out loud, no notes.
   If you can't teach it plainly, you don't know it yet.
5. **Papers are the source of truth.** Resources are mapped per phase in
   [RESOURCES.md](RESOURCES.md). Read the primary source where listed — secondary explainers
   quietly smooth over exactly the parts you need.

## The roadmap

| Phase | Topic | Duration | Folder |
|-------|-------|----------|--------|
| 0 | Mathematical foundations: calculus, linear algebra, probability, information theory | 1 week | [phase-00-math-foundations](phase-00-math-foundations/) |
| 1 | The training engine: gradient descent, optimizers, loss landscapes | 2 weeks | [phase-01-optimizers](phase-01-optimizers/) |
| 2 | CNNs & vision: convolution, architecture lineage, ResNet, ViT | 2 weeks | [phase-02-cnns](phase-02-cnns/) |
| 3 | NLP foundations: tokenization, embeddings, language modeling | 2 weeks | [phase-03-nlp-foundations](phase-03-nlp-foundations/) |
| 4 | RNNs: recurrence, hidden state, BPTT, vanishing gradients | 2 weeks | [phase-04-rnns](phase-04-rnns/) |
| 5 | LSTM & GRU: cell state, gates, why they fix recurrence | 1.5 weeks | [phase-05-lstm-gru](phase-05-lstm-gru/) |
| 6 | Seq2Seq, encoder–decoder & the attention patch | 1.5 weeks | [phase-06-seq2seq-attention](phase-06-seq2seq-attention/) |
| 7 | The Transformer: self-attention, multi-head, positional encodings (centerpiece) | 4 weeks | [phase-07-transformer](phase-07-transformer/) |
| 8 | Modern NLP: BERT, GPT, the pretrain→finetune paradigm | 3 weeks | [phase-08-pretrained-nlp](phase-08-pretrained-nlp/) |
| 9 | LLMs: pretraining, scaling laws, RLHF/DPO, LoRA, KV cache, RAG | 3 weeks | [phase-09-llms](phase-09-llms/) |
| 10 | Generative models & multimodal: VAE, GAN, diffusion, CLIP | 3 weeks | [phase-10-generative-multimodal](phase-10-generative-multimodal/) |
| Capstone | Written teaching artifacts + paper reproductions (on paper) | 2–4 weeks | [capstone](capstone/) |

Phases are cumulative. Attention is impossible without the encoder–decoder bottleneck that
motivated it; LSTM is impossible without BPTT. Don't skip ahead.

## How a phase works

```
Read primary sources  →  work the concept inventory  →  derivations on paper
        →  confront the misconceptions  →  cold quiz  →  next phase
   ↑                                                              ↓
   └──────── ask, in conversation, until every entry is defensible ◄┘
```

- `checklist.md` in each phase folder is your contract: concept inventory, derivations,
  thought experiments, misconceptions, and cold-quiz questions. Don't tick a box you can't
  defend out loud.
- `notes/` holds written theory notes — yours and ours — developed as each phase is covered.
- Track everything in [PROGRESS.md](PROGRESS.md).
- Book/paper/video map: [RESOURCES.md](RESOURCES.md).

## Starting point

You're coming in at intermediate level, so Phase 0 is a calibration week, not a remediation
one: skim, then go straight for the entries that feel slow. Speed on the math = where to
spend the real effort.
