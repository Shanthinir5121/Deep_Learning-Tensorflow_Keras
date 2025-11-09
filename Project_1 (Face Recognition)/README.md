Observations – CNN Face Recognition

1️) Image size vs Accuracy & Runtime:

For 32×32, 64×64, and 128×128 images, all models gave very high accuracy (≈ 98–100%).

Runtime increased sharply with image size —
32×32 took ~70 s, 64×64 took ~180 s, and 128×128 took ~900–1000 s.

So, bigger images improve detail but cost much more time.

2️) Useful number of convolution layers:

Accuracy remained similar after 2 or 3 convolution layers.

Extra layers (4 blocks) only increased runtime without clear accuracy gain.
- Hence, 2 convolution blocks are sufficient and efficient.
- Effect of Dropout (= 0.5):

With dropout = 0.5, training accuracy dropped slightly but test accuracy stayed almost same.

This means dropout helps prevent overfitting, but here data was already clean, so effect was small.
- Without dropout gave the highest accuracy, but with dropout gave more balanced generalization.

- Best configuration:

Image size: 128×128

Convolution blocks: 2

Dropout: 0.0

Accuracy: Train = 100%, Test = 100%

Runtime: ≈ 992 s
