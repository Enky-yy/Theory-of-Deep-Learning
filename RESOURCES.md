# Resources mapped to phases

Consume resources *in phase*, not front-loaded. Primary = read/watch it seriously;
Optional = deepen when a checklist entry stays shaky. Since this curriculum is
no-code, papers and derivations carry more weight than usual — go to the primary
source wherever one is listed.

## Phase 0 — Math foundations
- 3Blue1Brown: Essence of Linear Algebra + Essence of Calculus (intuition refreshers)
- Strang, *Introduction to Linear Algebra* — SVD chapters (or the MIT OCW lectures)
- Cover & Thomas, *Elements of Information Theory*, ch. 2 (entropy, KL, coding)
- Optional: Pilgrim, "What is the backpropagation algorithm?" — for the reverse-mode view

## Phase 1 — Optimizers
- Ruder, "An Overview of Gradient Descent Optimization Algorithms" (survey)
- Kingma & Ba, Adam (read the bias-correction derivation carefully)
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW — the whole point)
- Optional: Bottou, Curtis & Nocedal, "Numerical Optimization for ML";
  Li et al., "Visualizing the Loss Landscape" (and its follow-up critiques)

## Phase 2 — CNNs
- CS231n course notes (convolutional network chapters) — still the clearest treatment
- He et al., 2016, "Deep Residual Learning" (read the ImageNet-depth experiment)
- Dosovitskiy et al., 2020, "An Image is Worth 16x16 Words" (ViT)
- Optional: Ioffe & Szegedy, BatchNorm (and Santurkar et al., "How Does
  BatchNorm Help Optimization?" — the modern explanation);
  Liu et al., "A ConvNet for the 2020s" (ConvNeXt); Clark et al. on attention
  vs convolution

## Phase 3 — NLP foundations
- Jurafsky & Martin, *Speech and Language Processing* (free draft): tokenization,
  word-sense, embeddings chapters
- Mikolov et al., Word2Vec papers (both: architecture, then negative sampling)
- Pennington et al., GloVe
- Sennrich et al., 2016 (BPE for NMT); Kudo & Richardson, SentencePiece
- Optional: Bolukbasi et al., "Man is to Computer Programmer as Woman is to
  Homemaker?" (debiasing — and its critiques)
- Leonard, "A Brief History of Word Embeddings" (blog — good historical framing)

## Phase 4 — RNNs
- Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" (read,
  don't run — the char-level LM examples are the point)
- Goodfellow et al., *Deep Learning*, ch. 10 (recurrent nets — the BPTT math)
- Pascanu et al., 2013, "On the Difficulty of Training Recurrent Architectures"
- Optional: Bengio et al., 1994, "Learning Long-Term Dependencies with Gradient
  Descent" (the original vanishing-gradient paper)

## Phase 5 — LSTM & GRU
- Hochreiter & Schmidhuber, 1997 (original LSTM — read the constant-error-carousel section)
- Greff et al., 2015, "LSTM: A Search Space Odyssey" (what actually matters — 8 variants)
- Cho et al., 2014 (GRU — the seq2seq paper it appeared in)
- Optional: Olah, "Understanding LSTM Networks" (Colah's blog — the standard visual walkthrough)

## Phase 6 — Seq2Seq & early attention
- Sutskever et al., 2014, "Sequence to Sequence Learning" (read the length-limit discussion)
- Bahdanau et al., 2015 (additive attention)
- Luong et al., 2015 (effective attention — the multiplicative lineage)
- Papineni et al., 2002 (BLEU — read the actual definition)
- Optional: Li et al., 2016, "Beam Search Curse"; Ranzato et al. on exposure bias

## Phase 7 — Transformer
- Vaswani et al., 2017, "Attention Is All You Need" (read twice, first pass for
  the architecture, second for every design justification)
- Su et al., 2021, RoPE; Press et al., 2021, ALiBi; Shaw et al., 2018 (relative position)
- Ba, Kiros & Hinton, LayerNorm; Zhang & Sennrich, RMSNorm; Shazeer, GLU variants
- Clark et al., 2019, "What Does BERT Look At?" (head specialization);
  Elhage et al., 2021, "A Mathematical Framework for Transformer Circuits"
  (induction heads, residual stream — read after your own understanding is solid)
- Dao et al., 2022, FlashAttention; Shazeer, 2019, MQA; Ainslie et al., GQA
- Optional: Alammar's "Illustrated Transformer" (first-pass visual);
  the "Transformer Explainer" interactive (Georgia Tech) — good for building
  geometric intuition without writing code

## Phase 8 — Pretrained NLP
- Devlin et al., 2019, BERT; Radford et al., 2019, GPT-2 (the zero-shot thesis
  is in the *conclusion* — read it); Brown et al., 2020, GPT-3 (in-context learning)
- Raffel et al., 2020, T5 (the systematic-recipe paper)
- Wang et al., GLUE; Bowman & Dahl, "What Will it Take to Fix Benchmarking?"
- Optional: Liu et al., RoBERTa; Sanh et al., DistilBERT;
  Rogers et al., "Going Deeper with BERT" (a thorough analysis paper)

## Phase 9 — LLMs
- Kaplan et al., 2020 (scaling laws); Hoffmann et al., 2022, Chinchilla
- Ouyang et al., 2022, InstructGPT (the RLHF pipeline, stage by stage)
- Rafailov et al., 2023, DPO (do the derivation yourself first)
- Hu et al., 2021, LoRA; Dettmers et al., QLoRA; Kwon et al., vLLM (PagedAttention)
- Lewis et al., 2020, RAG; Wei et al., 2022 (emergence) vs Schaeffer et al.,
  2023 ("Are Emergent Abilities a Mirage?" — read both, they disagree)
- Wei et al., 2022, Chain-of-Thought; Khandelwal et al., 2020 (the
  retrieval-vs-long-context trade, "Generalization through Memorization")
- Optional: Brown et al. on scaling emergent abilities;
  Biderman et al., "Pythia" (open scaling study);
  Tay et al., "UL2" (objective survey)

## Phase 10 — Generative & multimodal
- Kingma & Welling, 2014 (VAE — derive the ELBO yourself first);
  Rezende et al., reparameterization
- Goodfellow et al., 2014 (GAN); Arjovsky et al., WGAN (the JS-divergence critique)
- Ho et al., 2020, DDPM (read the training-objective section twice);
  Song et al., score-based generative modeling;
  Rombach et al., 2022, latent diffusion (the cross-attention conditioning section)
- Radford et al., 2021, CLIP; Dhariwal & Nichol, 2021 (diffusion vs GAN)
- Optional: Sohl-Dickstein et al., 2015 (the original diffusion idea);
  Ramesh et al., DALL-E 2 (prior + decoder structure)

## Cross-cutting
- Karpathy, *Neural Networks: Zero to Hero* — the best watch-along for Phases
  3–7 (and his "Deep Dive into LLMs" talk for Phase 9). No code required to
  benefit: the derivations on the whiteboard are the value.
- Goodfellow, Bengio, Courville, *Deep Learning* (online free) — ch. 6 (deep
  nets), 9 (optimization), 10 (RNN), 14 (autoencoders)
- Lilian Weng's blog — clean, citation-rich syntheses (attention, prompt
  engineering, RAG, RLHF)
- 3Blue1Brown, "But what is a neural network?" series — for the geometric
  grounding of everything above
