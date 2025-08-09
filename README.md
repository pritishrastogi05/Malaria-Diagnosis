# Malaria-Diagnosis
<h2><u><b>Malaria Diagnosis using Deep Learning</h2></b></u>
This project focuses on building a deep learning model to detect the presence of the Plasmodium parasite in microscopic blood cell images, thereby diagnosing malaria.

<h3><b><u>Project Components and Capabilities</h3></u></b>
This notebook demonstrates various aspects of building and training a deep learning model for image classification, including:

<h3><b><u>Data Loading and Preprocessing</h3></b></u>
Loading the malaria dataset from TensorFlow Datasets.
Splitting the dataset into training, validation, and testing sets.
Implementing image resizing and rescaling.
Applying various data augmentation techniques such as random rotations, flips, contrast, and brightness adjustments to improve model robustness.
Exploring Mix-Up data augmentation for creating synthetic training examples.
Model Architecture
Building a Convolutional Neural Network (CNN) using the TensorFlow Functional API, which allows for flexible and dynamic model structures.
Demonstrating Model Subclassing to create reusable custom layers, such as the FeatureExtractor that encapsulates a sequence of convolutional and pooling layers.
Custom Components
The project incorporates custom implementations to extend the standard TensorFlow/Keras functionalities:

<ul>
<li>Custom Metrics: A CustomAccuracy class is implemented by inheriting from tf.keras.metrics.Metric. This allows for defining custom logic to calculate accuracy, potentially incorporating factors or other modifications beyond the standard accuracy calculation.</li>
<li>Custom Layers: Beyond the FeatureExtractor, the notebook outlines how to create a custom layer with trainable parameters by subclassing tf.keras.layers.Layer. This provides fine-grained control over the layer's behavior and allows for implementing novel layer types.</li>
<li>Checkpoints: The tf.keras.callbacks.ModelCheckpoint is used to save the model's progress during training. This enables saving the best performing model based on a monitored metric, resuming training from a saved state, and saving model checkpoints at specified intervals.</li>
<li>Hyperparameter Tuning:
Utilizing TensorBoard's HParams to set up and perform a grid search for hyperparameter tuning. This helps in finding the optimal values for parameters like learning rate, dropout rate, and regularization strengths to improve model performance.</li>
<li>Model Training and Evaluation:
Compiling the model with a suitable optimizer (Adam) and loss function (BinaryFocalCrossentropy) for binary classification.</li>
<li>Training the model using the prepared datasets.
Evaluating the model's performance using a comprehensive set of metrics including accuracy, precision, recall, AUC, true positives, true negatives, false positives, and false negatives.</li>
<li>Visualizing the training process and results, including confusion matrices and ROC curves.
Implementing a custom training loop to demonstrate how to train a model from scratch.</li>
Dependencies
</ul>

<P>
The project requires the following Python libraries:
<br>
<ul>
<li>tensorflow</li>
<li>tensorflow-probability</li>
<li>tf-keras</li>
<li>albumentations</li>
<li>wandb</li>
<li>tensorflow-estimator</li>
<li>matplotlib</li>
<li>numpy</li>
<li>pandas</li>
<li>scikit-learn</li>
<li>seaborn</li>
</ul>
</p>
