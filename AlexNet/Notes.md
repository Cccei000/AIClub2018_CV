### Background of the problem:
1. Small datasets cannot show the variability of the real world.
2. Larger datasets have been available.
3. Models with large learning capacity are needed.
4. Problem cannot be specified.  
-**what does it mean?**
5. Models should have much prior knowledge.

### Advantages of CNN (as mentioned):
1. Mutable depth and breadth.
2. Strong and correct assumptions.  
-**what are they?**
3. Fewer parameters and connections, easier to train.
4. Perhaps only slightly worse optimum.  
-**why?**
5. Facilitated training with GPUs.
6. No severe overfitting.

### Contributions of the paper:
1. Best results at that time with GPU implementation.
2. A number of new and unusual features to improve performance and time.
3. Techniques for preventing overfitting.
4. Discovery on an important depth.

### About the dataset:
1. ILSVRC: 1000 images in each of the 1000 classes.  
-**single-labeled or muti-labeled:** muti-labeled
> For each image, algorithms will produce a list of at most 5 object categories in the descending order of confidence. The quality of a labeling will be evaluated based on the label that best matches the ground truth label for the image. The idea is to allow an algorithm to identify multiple objects in an image and not be penalized if one of the objects identified was in fact present, but not included in the ground truth.  
> The ground truth labels for the image are gk, k=1,...,n with n objects labeled.   
> ![ILSVRC2010](http://image-net.org/challenges/LSVRC/2010/)
2. 2010: available test labels, 2012: unavailble test labels.
3. top-1, top-5.
4. Rescale images and crop out the centeral patch.  
-**does interpolation matter?**

### Architecture:
[architecture](https://github.com/Cei1ing/AIClub2018_CV/blob/master/AlexNet/Architecture.JPG?raw=true)

# Features:
### 1. ReLU:
* Two concepts: saturating nonlinearity, non-saturating nonlinearity.
* In terms of training time with gradient descent, saturating nonlinearities are much slower than the non-saturating nonlinearity Relu function f(x) = max(0; x).  
-**why?**
### 2. GPUs:
* The GPUs communicate only in certain layers.
* Some kernels only take input from kernels on the same GPU.  
-**why a problem for cross-validation?**  
-**what does it mean by independent columns?**  
### 3. Local Response Normalization:
* Loacl normaliztion scheme still aids gerneralization, despite ReLUs' desirable property.
* It normalizes the original output of each channel with several adjacent outputs of spatially corresponding position across channels.
* A form of lateral inhibition.  
-**what's this:** imitate the local competition between biological neurons, stress the higher response. 
* Resemble local contrast normalization.  
-**what's this:** subsract the weighted averge of adjacent region across channels and then divided by variance.
### 4. Overlapping Pooling:
* More difficult to overfit with overlapping pooling.  
-**why?**
