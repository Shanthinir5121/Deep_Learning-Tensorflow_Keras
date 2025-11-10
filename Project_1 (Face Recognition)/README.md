## Face_Recognition

This project focuses on Face Recognition using Convolutional Neural Networks (CNNs) trained on color face images extracted from real video clips captured under different lighting conditions. The workflow includes video-to-frame extraction, grayscale and binary conversion, face cropping, normalization, and resizing to build a clean dataset. The CNN model is trained on resized RGB images of sizes 32×32, 64×64, and 128×128 to analyze how image resolution and model complexity affect performance. Multiple experiments were performed by varying the number of convolution layers (2, 3, and 4) and applying dropout (0.5) to study overfitting and runtime differences.

Observations:

-> Image size vs Accuracy & Runtime:
    For 32×32, 64×64, and 128×128 images, all models gave very high accuracy (≈ 98–100%).
    Runtime increased sharply with image size —
    32×32 took ~70 s, 64×64 took ~180 s, and 128×128 took ~900–1000 s.
    So, bigger images improve detail but cost much more time.

-> Useful number of convolution layers:
    Accuracy remained similar after 2 or 3 convolution layers.
    Extra layers (4 blocks) only increased runtime without clear accuracy gain.
    Hence, 2 convolution blocks are sufficient and efficient.

-> Effect of Dropout (= 0.5):
    With dropout = 0.5, training accuracy dropped slightly but test accuracy stayed almost same.
    This means dropout helps prevent overfitting, but here data was already clean, so effect was small.
    Without dropout gave the highest accuracy, but with dropout gave more balanced generalization.
