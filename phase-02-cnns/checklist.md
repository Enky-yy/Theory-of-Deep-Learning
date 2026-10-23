# Phase 2 — CNNs & Vision (2 weeks)

Goal: convolution as a *mechanism* — what it computes, why parameter sharing is the
inductive bias, how the architecture lineage solved depth — plus the ViT pivot that
sets up Phase 7. You should be able to trace every major design decision to the
failure it addressed.

## Concepts

- Convolution arithmetic: kernel, stride, padding, dilation, groups; the output-size
  formula and why it's floor((H + 2p − k)/s) + 1
- Channels-as-filters: what a feature map's each entry *means* (activation of one
  learned detector at one location); depth = vocabulary of local patterns
- **Parameter sharing = the inductive bias**: a cat in the top-left and bottom-right
  activate the same filters; translation equivariance (move input → feature map moves
  identically) and where it breaks (striding destroys exact equivariance; pooling
  destroys location precision deliberately)
- Receptive fields: computing the effective field of a stack by hand; how depth,
  kernel size, stride, and dilation each grow it; the RF-vs-resolution tradeoff
  that motivated dilated convolutions
- Backprop through convolution: gradient w.r.t. input (needed for saliency/adversarial
  examples) vs w.r.t. kernel; the "sum over overlapping patches" structure
- Architecture lineage as a failure-log: LeNet (homage) → AlexNet (ReLU, dropout,
  GPUs made depth viable) → VGG (small kernels, depth) → GoogLeNet (multi-scale
  branches, 1×1 convs for channel reduction) → **ResNet** (the depth wall — 56-layer
  worse than 20-layer = optimization failure, not overfitting — solved by identity
  shortcut: gradient flows unimpeded, the "at least no worse" argument) → DenseNet
  (concatenated shortcuts) → EfficientNet (compound scaling)
- BatchNorm: what exactly is normalized (per-channel, per-batch statistics), the
  train/eval running-average split, why it stabilizes (smooths loss landscape,
  allows higher LR — the internal-covariate-shift explanation is *not* the accepted
  one anymore), why it fails at small batch sizes
- BN vs LayerNorm: batch statistics vs per-sample statistics — why transformers
  (variable-length sequences, batch of one at inference) chose LayerNorm. This is
  the hinge between vision and Phase 7.
- Modern convs: depthwise-separable convolutions (factorize spatial mixing from
  channel mixing → ~8–9× fewer FLOPs), MobileNet, ResNeXt (cardinality), ConvNeXt
  (modernizing convnets with transformer ingredients — evidence the ingredients,
  not the attention, do much of the work)
- **The conv-vs-attention debate**: CNN locality + weight sharing + hierarchy =
  strong sample efficiency, weak long-range modeling. Attention: global from layer 1,
  no built-in locality bias, data-hungry. ViT: patchify → linear embed → standard
  transformer; needs more data to discover locality by itself; wins at scale.
  DeiT, hierarchical ViTs (Swin) as the synthesis.

## Derivations (on paper)

- [ ] Output shape of conv(K×K, stride s, pad p) on H×W input; then the receptive
      field of a 2-conv stack (both 3×3, stride 1), then with a stride-2 conv between
- [ ] Gradient of the conv loss w.r.t. one kernel weight: which activations
      contribute, and why it's a sum over all spatial positions where that weight was used
- [ ] ResNet's core argument: write y = F(x) + x, then ∂L/∂x = ∂L/∂y · ∂y/∂x with
      ∂y/∂x = ∂F/∂x + I — the identity term guarantees the gradient never dies
- [ ] Parameter count comparison: 3×3 conv on (Cin→Cout) vs depthwise-3×3 + 1×1
      (Cin→Cout) — do the arithmetic, get the ratio for Cin=Cout=512

## Misconceptions to kill

- [ ] "ResNets work because deeper nets can now also learn identity" — a plain linear
      layer *can* represent identity; it just doesn't *find* it. The shortcut changes
      the optimization landscape (gradient highway), not the hypothesis class much.
- [ ] "BatchNorm is a regularization technique" — its original purpose is optimization
      stability; the regularization effect is a side effect (batch statistics noise)
      that people later relied on.
- [ ] "CNNs are translation invariant" — equivariant, not invariant (the feature map
      moves with the input; invariance comes later from pooling/stride/heads).
- [ ] "ViT is better than CNNs now" — at what data scale? Restate the comparison
      with dataset size on the axis every time.
- [ ] "1×1 convs do spatial mixing" — they mix *channels* at one location only;
      they're pointwise linear layers in disguise (channel MLP).

## Thought experiments

- [ ] You have 100 training images of cats. Predict the ranking (no ties) of:
      small CNN, large CNN, ViT, plain MLP — and give the mechanism for each position.
- [ ] A 56-layer plain net underperforms a 20-layer one on train *and* test.
      Overfitting or optimization failure? What single experiment discriminates, and
      what did ResNet's authors actually observe?
- [ ] At inference you must classify 32×32 patches, one at a time, on a device with
      no batch dimension support. What happens to BatchNorm? Why does this single
      fact push an entire field toward LayerNorm?
- [ ] Take the GoogLeNet inception module: why does 1×1 convolution before the 3×3
      and 5×5 branches cut compute so much? Do the channel arithmetic.

## Cold quiz
1. Compute output shape and receptive field growth for a 3-layer conv stack you specify.
2. Why does parameter sharing give translation *equivariance*? Name two operations in
   a typical CNN that break it, and say what is lost.
3. Write the residual gradient argument from memory — why can't the gradient vanish
   through the shortcut path?
4. What does BatchNorm normalize, in train vs eval mode, and why does small batch
   size break it?
5. BN vs LayerNorm: one formula-level difference each, and the architectural
   consequence for transformers.
6. CNN inductive bias vs ViT: one sentence each on sample efficiency, long-range
   modeling, and the regime where each wins.
