# Phase 10 — Generative Models & Multimodal (3 weeks)

Goal: the generative-model zoo from a single unifying question — *how do you model
p(x) when x is an image?* — and the multimodal bridge where text conditions vision
through the same attention machinery you now own.

## Concepts

### 10a. The framing: density, sampling, and the three families
- What "generative model" means precisely: learn p_θ(x) (or a usable surrogate)
  so you can *sample* and ideally *evaluate* likelihoods
- The taxonomy by what they give you: likelihood-based (VAE, flow, autoregressive),
  implicit (GAN), diffusion (hybrid: likelihood-motivated, sample-driven)
- Latent variables: why adding z helps (p(x) is hopelessly multimodal; a latent
  maps simple z → complex x; the manifold hypothesis as motivation)

### 10b. VAEs
- The encoder-decoder picture with the *inference* encoder: q_φ(z|x) approximates
  the intractable posterior p(z|x)
- **The ELBO derivation**: log p(x) ≥ E_q[log p(x|z)] − KL(q(z|x)‖p(z)) —
  the two terms read as reconstruction + compression/regularization; every step
  of the derivation on paper (Jensen; the importance-sampling identity trick)
- The reparameterization trick: z = μ + σ⊙ε, ε ~ N(0,1) — why it's needed
  (move the sampling out of the gradient path; stochastic node otherwise blocks
  backprop)
- The KL-collapse failure mode (posterior ignores z — reconstructions blurry);
  why blurry *in principle* (MSE likelihood = conditional mean over modes —
  connect Phase 0's Gaussian-MLE view of MSE)
- Forward KL (mode-covering) vs reverse KL (mode-seeking) — Phase 0's asymmetry,
  finally cashed in; β-VAE, VQ-VAE (discrete latents — the bridge to
  modern latent-token systems)

### 10c. GANs
- The minimax game: generator vs discriminator as binary classifier; the
  value function; the optimal-discriminator result (D* = p_data/(p_data + p_g))
  and what the generator's objective becomes under it (→ JS divergence at
  optimum — know the derivation's *shape*)
- Why GANs give sharp samples: implicit sampling, no likelihood; what they lose
  (no evaluable likelihood, no obvious conditioning control)
- Failure modes with mechanisms: mode collapse (generator finds one mode, D
  catches up, oscillation), training instability (the delicate equilibrium),
  divergence measures (Wasserstein-GAN: the Earth-mover distance, the Lipschitz
  constraint, gradient penalty — why WGAN fixed the *loss signal* not the game)
- The historical role: StyleGAN-quality faces drove the 2010s; Diffusion displaced
  them ~2020 for the reasons the next section makes inevitable

### 10d. Diffusion models
- The two processes, stated cleanly: **forward** (gradually add Gaussian noise
  over T steps until x_T ~ N(0,I) — a fixed, known corruption schedule);
  **reverse** (learn to denoise: predict the noise ε at each noise level)
- The DDPM training objective: L = E[‖ε − ε_θ(x_t, t)‖²] — the stunning simplicity
  (regression on known noise); why MSE here is *not* the blurry-MSE of VAEs
  (predicting noise, not pixels; the model predicts a *score*, one step at a
  many-step process)
- Why diffusion won: stable training (regression, not a game), mode coverage,
  the sampler-quality trade-off (steps vs quality; DDIM as the deterministic
  sampler; distillation/few-step methods — consistency models, at concept level)
- **Latent diffusion (Stable Diffusion's actual architecture)**: run diffusion in
  a VAE-compressed latent space (8× cheaper than pixel space — do the
  arithmetic); the U-Net with cross-attention conditioning; **this is where
  Phase 7 closes a loop**: the text conditioning *is* cross-attention, Q from
  image latents, K/V from the text encoder's tokens; classifier-free guidance
  (the CFG formula; why guidance trades diversity for adherence; what
  "guidance scale 7.5" does to the effective distribution)
- Score-based view: ε-prediction = learning the score function ∇log p; the
  connection to score matching; the Langevin/sampling-step story at concept level

### 10e. Multimodal & vision-language
- **CLIP**: the contrastive objective (image-text batch, N² similarity matrix,
  InfoNCE — write the loss); zero-shot classification by prompt templating
  ("a photo of a {class}"); why it generalizes (400M pairs; web data as the
  supervision); the embedding-space lessons (Phase 3's distributional hypothesis,
  now across modalities)
- Contrastive learning generally: SimCLR/MoCo ideas at concept level (what makes
  negatives necessary; the false-negative problem) — one section, not deep
- Image-text generation: the captioner family (encoder image → decoder text) and
  the modern unified models (Chameleon-style early fusion vs cross-attention
  conditioning — two architectures for "see and speak")
- Where the field is heading conceptually: unified multimodal tokens (images as
  tokens in the same sequence — Phase 7's architecture, generalized), the
  modality-agnostic-LM framing; tokenization of images (VQ tokens, patches) in
  the Phase 3 light (every modality needs its "text becomes numbers" step)

## Derivations (on paper)

- [ ] The ELBO: full derivation from log p(x), every step; then read both terms
      off the final line with plain-language labels
- [ ] Reparameterization: show why E_{z~q}[f(z)] with z = μ+σε has gradients
      through μ and σ, whereas the direct-sampling formulation doesn't
- [ ] GAN: from the minimax value function, show that optimal D is
      p_data/(p_data+p_g), then substitute back to get the JS-divergence form
- [ ] DDPM: the forward process as a closed form (x_t in terms of x_0 and ε —
      the ᾱ_t notation), then the reverse-training objective's derivation at
      the level of "MSE on ε is equivalent to a weighted denoising score match"
- [ ] Latent diffusion compute: 64×64×3 pixel space vs 8×8×4 latent (VAE ×8
      downsample) — the per-step compute ratio, and why it makes consumer-GPU
      training possible
- [ ] CLIP's InfoNCE loss: the N² logits, both diagonal directions, the
      temperature's role; then the zero-shot classification procedure exactly
- [ ] CFG: the guided-score formula, and its interpretation as a shift along the
      conditional-unconditional difference (why higher scale = sharper but
      less diverse)

## Misconceptions to kill

- [ ] "VAEs generate blurry images because MSE is bad" — MSE under a unimodal
      likelihood forces conditional averaging; the fix (better likelihoods,
      latent diffusion) addresses the *structure*, not the loss weight.
- [ ] "Diffusion models are autoencoders" — the reverse network is trained by
      plain noise-prediction regression; the 'modeling' of p(x) emerges across
      the whole sampling trajectory, not in one reconstruction.
- [ ] "GANs are dead" — displaced for image synthesis, alive where no likelihood
      or exact sampling matters (super-resolution variants, some training-free
      setups, the WGAN-GP machinery in regularization)
- [ ] "CFG makes the model better" — it makes it more *condition-adherent* at the
      cost of diversity and realism at high scale; it's a sampler-time knob, not
      a training improvement.
- [ ] "CLIP understands images and text jointly" — it learns a shared *embedding
      space* by matching; compositionality failures (attribute binding — "red
      cube, blue sphere" vs swapped) are well documented; the understanding is
      retrieval-shaped.
- [ ] "Image tokens are just another tokenizer" — patches-as-continuous-vectors
      (ViT) vs VQ-discrete tokens (DALL-E 2 era, Chameleon) are different choices
      with different consequences (continuous = diffusion-friendly; discrete =
      autoregressive-LM-friendly).

## Thought experiments

- [ ] A museum wants an exhibit where visitors describe an artwork and see it
      generated. Using latent diffusion + CLIP only: describe the full pipeline,
      every model's role, and the two most likely visitor-visible failure modes.
- [ ] Same face, 1000 samples from a GAN vs 1000 from a diffusion model: which
      set shows mode collapse, and what's the observable signature? Why can't
      VAEs produce either failure mode? (Answer from each family's objective.)
- [ ] You must generate 512×512 images in under 50ms on a phone. Which family
      and which specific techniques (steps? distillation? latent space?) — and
      what quality axis do you knowingly sacrifice?
- [ ] CLIP's embedding puts "a photo of a dog" and "a photo of a wolf" close.
      Name one application where this is dangerous and one where it's useful.

## Cold quiz
1. The ELBO: derive it, label both terms, and say what each term *does* to the
      latent space.
2. Reparameterization trick: what breaks without it, in one gradient-shaped sentence.
3. GAN at optimal D: the substitution that reduces the game to a divergence —
      name the divergence, then WGAN's fix and what it fixed.
4. DDPM training: the objective (one line), why predicting ε avoids VAE blur,
      and the forward-process closed form.
5. Latent diffusion: the compute argument (with the ×8 arithmetic), and where
      cross-attention enters (Q from where, K/V from where — Phase 7 connection).
6. CLIP: InfoNCE in one line, the zero-shot procedure, one documented
      compositionality failure.
7. CFG formula + what the scale knob trades.
