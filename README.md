# Pet Classification Tensorflow Model Using CNN

## Project Objective
To build a CNN model that classifies the given pet images correctly into dog and cat images.

The project scope document specifies the requirements for the project “Pet Classification Model Using CNN.” Apart from specifying the functional and nonfunctional requirements for the project, it also serves as an input for project scoping.

## Project Description and Scope
We are provided with a collection of images of pets (cats and dogs) of different sizes with varied lighting conditions. The goal is to build a CNN image classification model using TensorFlow that trains on this data and calculates the accuracy score on the test data.

## Project Guidelines
The CNN model (`cnn_model_fn`) should have the following layers:
- Input layer
- Convolutional layer 1 with 32 filters of kernel size [5,5]
- Pooling layer 1 with pool size [2,2] and stride 2
- Convolutional layer 2 with 64 filters of kernel size [5,5] and stride 2
- Pooling layer 2 with pool size [2,2] and stride 2
- Dense layer whose output size is fixed in the hyper parameter: `fc_size=32`
- Dropout layer with dropout probability 0.4

The class prediction is done by applying a softmax activation on the output of the dropout layer.

The training step involves defining the loss function and minimizing it. For the evaluation step, the accuracy is calculated.

The program should be run for 100, 200, and 300 iterations, respectively, followed by a report on the final accuracy and loss on the evaluation data.

## Model Architecture

The CNN model implemented in this notebook follows the specifications outlined in the project guidelines. It consists of the following layers:

1.  **Input Layer:** The input layer receives the image data. Since the images are grayscale and resized to 64x64 pixels, the input shape is (64, 64, 1).
2.  **Convolutional Layer 1:** This layer applies 32 filters of size [5, 5] with a ReLU activation function. Convolutional layers learn features from the input images by convolving the filters over the image data.
3.  **Pooling Layer 1:** This layer performs max pooling with a pool size of [2, 2] and a stride of 2. Pooling layers reduce the spatial dimensions of the feature maps, helping to reduce computation and control overfitting.
4.  **Convolutional Layer 2:** This layer applies 64 filters of size [5, 5] with a ReLU activation function.
5.  **Pooling Layer 2:** This layer performs max pooling with a pool size of [2, 2] and a stride of 2.
6.  **Flatten Layer:** This layer flattens the output of the pooling layers into a 1D vector, which can be fed into the dense layers.
7.  **Dense Layer:** This is a fully connected layer with 32 units and a ReLU activation function. Dense layers learn global patterns in the data. The output size is determined by the `fc_size` hyperparameter.
8.  **Dropout Layer:** This layer applies dropout with a probability of 0.4. Dropout is a regularization technique that randomly sets a fraction of the input units to zero during training, which helps prevent overfitting.
9.  **Output Layer:** This is a dense layer with 2 units (for the two classes: cat and dog) and a softmax activation function. The softmax function outputs a probability distribution over the classes, indicating the model's confidence in each class prediction.

The model is compiled using the Adam optimizer and the categorical crossentropy loss function, which is suitable for multi-class classification problems. The accuracy metric is used to evaluate the model's performance during training and evaluation.

## Data Preparation

The image data is loaded from the specified directories and preprocessed for training and evaluation. The preprocessing steps include:

*   Resizing the images to a fixed size (64x64 pixels).
*   Converting the images to grayscale.
*   Labeling the images using one-hot encoding (e.g., [1, 0] for cats and [0, 1] for dogs).
*   Saving the preprocessed images and labels as NumPy arrays.

Data augmentation is applied to the training data using `ImageDataGenerator` to increase the size and diversity of the training set. This includes random rotations, shifts, shear transformations, zoom, and horizontal flips.

The `flow_from_directory` method of `ImageDataGenerator` is used to create data generators that load images in batches and apply the specified augmentations.

## Link to the project

[[ Link Here](https://colab.research.google.com/drive/1mAYYViFa3dEFcLXJjgdY2LL_TL4B4IXS?usp=sharing)]
