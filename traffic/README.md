# Traffic Sign Recognition: Model Design

This document outlines the process of designing and tuning the convolutional neural network (CNN) for the Traffic Sign Recognition project, reflecting the final performance results.

## Initial Model Architecture

The goal was to create a model that could accurately classify images of traffic signs into 43 distinct categories. Given that this is an image classification task, a Convolutional Neural Network (CNN) was the natural choice.  The initial architecture was designed to be a common but effective baseline.

The starting model consisted of:
1.  A Convolutional Layer with 32 filters of size (3, 3) and a ReLU activation function to identify basic features like edges.
2.  A Max-Pooling Layer with a (2, 2) pool size to downsample the feature map, reducing computational load.
3.  A Flatten Layer to convert the 2D feature maps into a 1D vector.
4.  A Dense Hidden Layer with 128 units and a ReLU activation function.
5.  A Dropout Layer with a rate of 0.5 to prevent overfitting.
6.  A final Dense Output Layer with 43 units (one for each category) and a `softmax` activation function to output a probability distribution.

## Experimentation and Final Results

The initial single-layer model performed reasonably well, but to achieve higher accuracy, further experimentation was necessary.

Adding Network Depth: The most significant improvement came from adding a second `Conv2D` and `MaxPooling2D` pair to the network. The new convolutional layer was given 64 filters, allowing the model to learn more complex and abstract features from the initial patterns. This architectural change proved to be highly effective.

Final Performance: After 10 epochs of training, the model's performance was evaluated on the test set. The model achieved a final test accuracy of 95.05% with a loss of 0.1794. This result confirms that the deeper architecture was successful in learning the distinguishing features of the traffic signs.

Analysis of Training: An interesting observation from the training process is that the final training accuracy at the end of epoch 10 was approximately 84%, which is lower than the final test accuracy. This is a positive indicator and is likely due to the dropout layer. Dropout is active during training (handicapping the model to force it to generalize) but is inactive during evaluation. The fact that the model performed better on unseen test data demonstrates that the dropout layer was highly effective at regularizing the model and preventing overfitting.