## 14. Semantic Segmentation and U-Net

### Semantic Segmentation

The task of assigning a class label to every pixel in an image. Produces a fine-grained label map (e.g., road, building, person, sky). It is about classes, not instances — two people share the same label "person." Requires both low-level detail (precise boundaries) and high-level context (class identity).

**FCN (Fully Convolutional Network)** — Long et al. (2015) introduced the foundational architecture for semantic segmentation: replace all fully-connected layers in a classification CNN with convolutional layers, enabling pixel-wise output. FCN pioneered the idea of skip connections from earlier layers to improve spatial precision. U-Net builds directly on these FCN concepts.

### U-Net Architecture

An **encoder–decoder** architecture with **skip connections**, shaped like a "U":

- **Encoder (contracting path):** Successive conv layers + pooling reduce spatial resolution while capturing context.
- **Decoder (expansive path):** Upsamples features back to original resolution for pixel-wise predictions.
- **Skip connections:** Feature maps from the encoder are copied to the decoder at corresponding scales, providing high-resolution details lost during pooling.

Each encoder stage: two 3×3 convs + 2×2 pooling. Each decoder stage: upsampling + two 3×3 convs. Final 1×1 conv produces the segmentation map.

### Unpooling, Upsampling, and Transposed Convolution

**Unpooling (Max-Unpooling):** Uses stored max-pooling indices to place values back at original locations, filling others with zeros ("bed of nails" pattern). Not learnable but preserves exact positions of important features. Usually followed by a conv layer.

**Transposed Convolution (Learned Upsampling):** A learnable upsampling layer. Inserts zeros between input positions, then applies a learned convolution filter. Can learn task-specific interpolation. More flexible than fixed interpolation. Risk: checkerboard artifacts if kernel/stride don't align well.

**Upsampling + Convolution:** Simple upsampling (nearest neighbor or bilinear) followed by a standard convolution. Often avoids checkerboard artifacts. Functionally similar to transposed convolution.