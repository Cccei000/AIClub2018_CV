### Background of the problem:
1. Small datasets cannot show the variability of the real world.
2. Larger datasets have been available.
3. Models with large learning capacity are needed.
4. Problem cannot be specified even by large datasets.
5. Models should have much prior knowledge.

### Advantages of CNN (as mentioned):
1. Mutable depth and breadth.
2. Strong and correct assumptions:  
   1. Stationarity of statistics: the statistics of one part of the image are the same as any other part.
   2. Locality of pixel dependencies.
3. Fewer parameters and connections, easier to train.
4. Facilitated training with GPUs.

### Disadvantages of CNN (as mentioned):
1. Overfitting.
2. Sensitive to depth.
3. Better GPU needed.

### Contributions of the paper:
1. Best results of the day with GPU implementation.
2. A number of new and unusual features to improve performance and time.
3. Techniques for preventing overfitting.

### About the dataset:
1. [ILSVRC2010](http://image-net.org/challenges/LSVRC/2010/): 1000 images in each of the 1000 classes, one ground truth label per image.
2. 2010: available test labels, 2012: unavailble test labels.
3. top-1, top-5.
4. Rescale images, crop out the centeral patch, and ***subtracting the mean activity***?

### Architecture:
![architecture](https://github.com/Cei1ing/AIClub2018_CV/blob/master/Paper/AlexNet.JPG?raw=true)
* The kernels of the second, fourth, and fifth convolutional layers are connected only to kernel maps in the previous layer on the same GPU.
* Response-normalization layers follow the first and second convolutional layers. 
* Max-pooling layers follow response-normalization layers and the fifth convolutional layer.
* ReLU non-linearity is applied to the output of every learnable layer.  
* Dropout in the first two fully-connected layers.  
-**why the architecture above?**  
-**how to improve it?**
* Softmax produces the final distribution.
* ***Maximizes the multinomial logistic regression objective?***

# Features:
### 1. ReLU:
* Two concepts: saturating nonlinearity, non-saturating nonlinearity.
* In terms of training time with gradient descent, saturating nonlinearities are much slower than the non-saturating nonlinearity Relu function f(x) = max(0; x).  
-saturating nonlinearities like sigmoid and tanh are too smooth for gradient descent, leading to gradient vanishing.

### 2. GPUs:
* The GPUs communicate only in certain layers.
* Some kernels only take input from kernels on the same GPU.  
-**why is this a problem for cross-validation?**  
-**what does it mean by independent columns?**

### 3. Local Response Normalization:
* Loacl normaliztion scheme still aids gerneralization, despite ReLUs' desirable property.
* It normalizes the original output of each channel with several adjacent outputs of spatially corresponding position across channels.  
-**why does this work for AlexNet but fail in VGGNet?**
* A form of lateral inhibition.  
-imitate the local competition between biological neurons, stress the higher response. 
* Resemble local contrast normalization.  
-subsract the weighted averge of adjacent region across channels and then divided by variance.  
* Applied after certain layers' RuLU.

### 4. Overlapping Pooling:
* More difficult to overfit with overlapping pooling.  
-improve the ability to extract features. 

### 5. Data Augmentation:
* To reduce overfitting, generated on the CPU, computationally free.
* Scheme one: train on randomly extracted patches and their horizontal reflections, test on certain patches and averge the softmax predictions.
* Scheme two: perform PCA on the set of RGB pixel values throughout the training set.

### 6. Dropout:
* Set to zero the output of each hidden neuron with a certain probability.
* Reduces complex co-adaptations of neurons, learn more robust features.
* The paper just multiplies the outputs of all neurons by 0.5 to approximate the geometric mean of various dropout networks.  
-**why is this reasonable?**

### 7. Learning:
* SGD, batch size 128, momentum 0.9, weight decay 0.0005.
* ***Weight decay here is not just a regularizer, it reduces the model's training error?***
* Learning rate 0.01, divided by 10 when validation error rate stops improving.

# Improvements:
Referring to ZFNet:  
* Smaller filters in conv1: cover mid frequencies.
* Smaller strides in conv1: reduce overlapping.
* Depth of the model is important for obtaining good performance.
* Increasing the size of the middle convolution layers gives a gain in performance.
* But increasing these and enlarging the fully connected layers results in over-fitting.
* The model is truly localizing the objects within the scene.
* A different loss function that permits multiple objects per image may improve performance.
