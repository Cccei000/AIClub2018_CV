### Background of the problem:
1. Small datasets cannot show the variability of the real world.
2. Larger datasets have been available.
3. Models with large learning capacity are needed.
4. Problem cannot be specified. -**what does it mean?**
5. Models should have much prior knowledge.

### Advantages of CNN (as mentioned):
1. Mutable depth and breadth.
2. Strong and correct assumptions. -**what are they?**
3. Fewer parameters and connections, easier to train.
4. Perhaps only slightly worse optimum. -**why?**
5. Facilitated training with GPUs.
6. No severe overfitting.

### Contributions of the paper:
1. Best results at that time with GPU implementation.
2. A number of new and unusual features to improve performance and time.
3. Techniques for preventing overfitting.
4. Discovery on an important depth.

### About the dataset:
1. ILSVRC: 1000 images in each of the 1000 classes.
2. 2010: available labels, 2012: unavailble labels.
3. top-1, top-5.
4. Rescale images and crop out the centeral patch. -**does interpolation matter?**

### Architecture:
![architecture](https://github.com/Cei1ing/AIClub2018_CV/blob/master/AlexNet/Architecture.JPG?raw=true)

# Features:
### 1. ReLU:
* Two concepts: saturating nonlinearity, non-saturating nonlinearity.
* In terms of training time with gradient descent, saturating nonlinearities are much slower than the non-saturating nonlinearity Relu function f(x) = max(0; x). -**why?**
### 2. GPUs:
* The GPUs communicate only in certain layers.
* Some kernels only take input from kernels on the same GPU.
* -**why a problem for cross-validation?**
### 3. Local Response Normalization:
* Loacl normaliztion scheme still aids gerneralization, despite ReLUs' desirable property.


