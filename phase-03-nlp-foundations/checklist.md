# Phase 3 — NLP Foundations & Neural Language Modeling I (2 weeks)

Goal: the layer most courses skip and it shows. Text must become numbers before any
model touches it — you should understand exactly what is lost and gained at every
step of that conversion, and why embeddings changed NLP forever.

## Concepts

- **Tokenization, in depth**: why word-level tokenization fails (vocab explosion,
  OOV/UNK, morphology blindness); char-level's opposite tradeoff (tiny vocab,
  long sequences); **subword** as the resolution — BPE (byte-pair encoding: iteratively
  merge the most frequent adjacent pair), Unigram LM tokenization (probabilistic,
  Viterbi segmentation), WordPiece (BPE variant scoring by likelihood, not frequency)
- Implement BPE *by hand on paper* on a tiny corpus — the merges table, encoding
  an unseen word, why `ing` can be a token; special tokens ([PAD], [UNK], [CLS],
  [SEP], [MASK]) and why each exists
- Tokenization's downstream shadow: perplexity comparisons across different
  tokenizers are meaningless (different alphabets → different per-token difficulty);
  why multilingual models tokenize non-English text inefficiently; emoji/byte-level
  fallback (GPT-2's byte-level BPE) and why it killed the UNK problem
- **The one-hot curse**: vocab 50k → 50k-dim sparse vectors; no graded similarity
  (all pairs equidistant); the embedding layer as a learned lookup = a dense
  linear layer with one-hot input (equivalence worth stating precisely)
- **Distributional semantics**: "you shall know a word by the company it keeps"
  (Firth 1957) — the hypothesis that makes embeddings possible
- **Word2Vec**: CBOW vs skip-gram; the actual objective (predict neighbor from center
  or vice versa); why the naive softmax over 50k vocab is intractable → **negative
  sampling** (the key realization: it's a logistic-regression objective distinguishing
  real pairs from noise pairs; noise distribution ∝ unigram^¾ and why ¾)
- **GloVe**: log-co-occurrence matrix factorization; the explicit weighting function;
  how it differs from Word2Vec's "local context" view (global statistics)
- fastText: subword embeddings — handles OOV by composing unseen words from character
  n-grams; the reason morphologically rich languages benefit
- **Embedding geometry**: linear substructure (king − man + woman ≈ queen) — what
  exactly the analogy task measures; cosine vs Euclidean similarity; PCA/2D projection
  caveats (hubness, projection distortion); intrinsic (analogy, similarity) vs
  extrinsic (downstream task) evaluation
- Bias in embeddings: historical/occupational associations (he−doctor, she−nurse);
  the debiasing pipeline (Bolukbasi et al.: identify gender subspace, neutralize,
  equalize) and its limitations — bias in, bias out
- **Neural language modeling**: the Bengio 2003 setup (predict next token from
  previous n via embeddings + MLP); perplexity as the objective's exponentiated loss;
  train/val/test splits on *sequences* (split by document, never by token — the
  leakage trap)
- Recurrent LMs preview: why the fixed-n window is the bottleneck every later
  architecture fights (straight line to Phases 4–7)

## Derivations (on paper)

- [ ] BPE by hand: 10-word corpus, run 5 merge iterations, encode a held-out word,
      show the OOV handled via subwords
- [ ] Word2Vec skip-gram + negative sampling loss: write the sigmoid objective for
      one positive pair + k negatives; take the gradient w.r.t. the center word's
      embedding and interpret each term
- [ ] Show the embedding layer (nn.Embedding) ≡ one-hot input × weight matrix
- [ ] Softmax over 50k classes: cost of exact normalization vs negative sampling's
      O(k) — make the comparison concrete

## Misconceptions to kill

- [ ] "Embeddings encode meaning" — they encode *distributional co-occurrence
      structure*. The mapping from "distributional" to "semantic" is an empirical
      discovery, not a guarantee; antonyms co-occur heavily (good/bad).
- [ ] "The analogy task proves embeddings understand relations" — the linear
      offsets work best for the most frequent, most stereotyped relations; performance
      degrades sharply on infrequent and non-linear relations.
- [ ] "BPE merges are linguistic morphology" — they're frequency artifacts; `playing`
      may split as `play`+`ing` or `pl`+`aying` depending on corpus statistics.
- [ ] "Perplexity lower = model better" — only with identical tokenization and
      identical test data; state both conditions whenever you cite a number.
- [ ] "Negative sampling is an approximation of softmax" — it's a different
      objective (noise-contrastive); the embeddings still work because the
      *relative* geometry is preserved.

## Thought experiments

- [ ] You must build a sentiment classifier for a domain with 5% OOV rate with
      word-level, char-level, BPE-subword, and fastText embeddings. Rank them and
      justify with the mechanisms above.
- [ ] An English embedding analogy test scores 75%. Predict the score for German
      and for Japanese before looking anything up; then explain the actual numbers
      through tokenization granularity.
- [ ] Two embedding spaces trained on different corpora: identical vocab, different
      geometry. Is the analogy task still valid within each? Across? What would
      break if you mixed vectors from both in one model?

## Cold quiz
1. BPE vs Unigram vs WordPiece: the merge/score rule, one sentence each, and one
   downstream consequence of each difference.
2. Why does exact softmax over a large vocab cost O(V) per training example, and
   why does negative sampling make it O(k)? What objective is it *actually* optimizing?
3. Word2Vec vs GloVe: what statistic does each exploit? Where does each fail?
4. What does the linear analogy structure measure, and name two relation types where
   it's weak?
5. Why is split-by-token leakage? Construct a concrete scenario.
6. Give the full argument chain from "distributional hypothesis" to "embeddings
   encode useful semantics" — name every empirical step.
