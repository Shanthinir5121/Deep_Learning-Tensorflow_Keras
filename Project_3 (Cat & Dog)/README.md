## Dogs vs Cats Classification

This project uses a Convolutional Neural Network (CNN) to classify images of dogs and cats from a structured dataset containing separate training and testing folders. The model was trained and evaluated on both the full dataset and a reduced dataset consisting of randomly selected 250 cat images and 250 dog images, reducing the data by 8×.

The results showed a clear difference in performance:

-> Full dataset accuracy: ~78.15%

-> Reduced dataset accuracy: ~63.33%

-> Observation: Reducing the training data significantly decreases accuracy since the model is exposed to less feature diversity.

Conclusion: Accuracy does not remain the same when the dataset size is reduced — more data leads to better and more stable model performance.
