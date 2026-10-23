# Phase 0 — Mathematical Foundations (1 week)

Calibration, not remediation. You know DL at an intermediate level — this week finds
the cracks in the mathematical floor everything else stands on. Work fast through what
feels automatic; slow down the moment something costs effort. That slowness is data.

Companion notes: [../notes/math-refresher.md](../notes/math-refresher.md)

## Concepts (calibrate, don't relearn)

- [ ] Matrix calculus: chain rule in vector/matrix form; why backprop is one reverse
      sweep of local Jacobian products; shapes of gradients (∂L/∂W has W's shape)
- [ ] Linear algebra fluency: matrix-as-map, eigenvectors, SVD as rotate→scale→rotate,
      spectral vs Frobenius norms, PSD matrices, the geometry of the dot product
- [ ] Probability: joint/marginal/conditional, expectation, variance/covariance,
      Bayes' rule, common distributions and when each appears in DL
- [ ] Information theory: entropy as expected surprisal; cross-entropy; KL divergence
      (asymmetric — and why that asymmetry resurfaces in VAEs); perplexity as
      geometric-mean branching factor
- [ ] MLE view of losses: why cross-entropy training *is* maximum likelihood;
      why MSE training is Gaussian-likelihood MLE
- [ ] Convexity: convex sets/functions, Jensen's inequality, why deep nets being
      non-convex is survivable (saddle points vs local minima in high dimension)

## Derivations (on paper, from memory the second time)

- [ ] ∂(softmax + CE)/∂z = ŷ − y (the collapse — do it via the Jacobian, then
      via the log-sum-exp shortcut; know both)
- [ ] Backprop through a 2-layer MLP: every gradient, with shapes
- [ ] The Jacobian-product form of the chain rule for a composition of 3 functions
- [ ] Why the gradient of ‖W‖²_F is 2W, and what gradient descent with that term does
      to W in one step (connect to weight decay in Phase 1)

## Misconceptions to kill

- [ ] "KL divergence is a distance" — it's asymmetric and can exceed 1; find the
      direction where it's infinite while the other is finite (delta distributions)
- [ ] "Low training loss means the model understood the data" — CE loss equals
      negative log-likelihood; a model can achieve low CE via calibration tricks
      without any structure
- [ ] "Softmax is nonlinear in a way that matters for scaling" — softmax is
      shift-invariant; adding a constant to all logits changes nothing. Why, and
      where that fact is used (numerical stability everywhere)

## Thought experiments

- [ ] A model outputs probabilities [0.9, 0.05, 0.05] for the true class 0. Compute its
      CE loss. Now the true class is 1 and it outputs [0.05, 0.9, 0.05]. Same loss.
      What does a *single* loss value actually tell you?
- [ ] Perplexity 20 vs perplexity 50 on two test sets — can you directly compare the
      models? What must be true of the test sets? (Tokenization ties in, Phase 3.)

## Cold quiz
1. Derive the softmax+cross-entropy gradient, both routes.
2. In one sentence each: entropy, cross-entropy, KL — then show CE = H + KL.
3. Why does one reverse sweep produce every gradient? What's the computational cost
   relative to the forward pass?
4. Spectral norm vs Frobenius norm: what does each measure, and name one DL use of each.
5. If perplexity is e³, what does the model "see" at each step?
6. Write the exact gradient update that weight decay performs, and (foreshadowing)
   where Adam will break the equivalence between this and L2 regularization.
