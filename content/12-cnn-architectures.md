## 12. CNN Architectures (AlexNet, Inception, ResNet), Depth vs. Width, and Batch Normalization

### CNN Principles

CNNs use local receptive fields and **weight sharing**. Filters slide across the image, reducing parameters and providing translation invariance. Pooling subsamples feature maps for translational tolerance and reduced size. Stacking conv + pooling layers builds a hierarchy of features: edges → parts → objects.

### AlexNet (2012)

- 5 convolutional layers + 3 fully-connected layers (8 learnable layers total)
- ~60M parameters
- Key innovations: ReLU activation, dropout in FC layers, GPU training
- First conv: 11×11 with stride 4, 96 kernels
- Max-pooling with overlap (3×3, stride 2)
- Local Response Normalization (LRN)
- Top-5 ImageNet error: ~16% (runner-up was ~26%)

### GoogleNet / Inception v1 (2014)

- 22 layers but only **~5–6.8M parameters** (the paper claims "12× fewer than AlexNet"; detailed layer analysis yields ~6.8M)
- **Inception modules:** parallel convolutions with multiple filter sizes (1×1, 3×3, 5×5) + pooling, concatenated
- 1×1 convolutions as bottlenecks to reduce dimensionality
- Global average pooling instead of large FC layers
- Top-5 accuracy: ~89%

### ResNet (2015)

- 50–152 layers with **residual blocks**: learn $F(x) = H(x) - x$, add input via skip connection
- Skip connections provide gradient highways → solve vanishing gradient problem
- Additional layers can fall back to identity mapping → depth never hurts
- Uses BatchNorm after each conv
- ResNet-152: ~94% top-5 accuracy

### Deeper vs. Wider Networks

Deeper networks represent more complex functions by hierarchical composition — each layer reuses earlier computations. Width increases parallel processing but may learn redundant features. Depth has been the more critical driver of improvements.

### Batch Normalization

Normalizes activations for each mini-batch: subtract batch mean, divide by batch standard deviation, then scale/shift with learnable parameters $\gamma$ and $\beta$. Benefits: allows higher learning rates, acts as mild regularization, mitigates vanishing/exploding gradients, makes networks less sensitive to initialization.

> [!warning] Disputed mechanism
> The original Ioffe & Szegedy (2015) paper claimed BN works by reducing "internal covariate shift." However, Santurkar et al. (NeurIPS 2018) demonstrated that BN's actual benefit is **smoothing the optimization landscape** (reducing the Lipschitz constant of the loss and gradients), making optimization more predictable and stable — not stabilizing input distributions. On an exam, mention both views.