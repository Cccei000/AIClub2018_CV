### Purpose:
* Improve the orginal architecture of AlexNet in the aspect of its depth.

### Architecture:
1. Image preporcessing: subtract the mean RGB value.
2. Convolutional layers with filters of size 3\*3, stride of 1 and SAME padding.
   1. 3\*3 filters and stride of 1 across the network (where AlexNet uses 11\*11 and 4, ZFNet uses 7\*7 and 2).  
   2. two 3\*3 conv.layers VS one 5\*5 conv.layer, three 3\*3 conv.layers VS one 7\*7 conv.layer.
   3. incorporate 3 ReLU, and decrease parameters.
3. 1\*1 filters with ReLu to increase the non-linearity of the decision function without other affects.
4. 5 max-pooling layers with window of size 2\*2 and stride 2. 
5. Two 4096 units FC layers, one 1000 units FC layer (for ILSVRC) and a soft-max layer.
6. ReLU where needed.
7. No local response normalisation - it increases memory and time.

### Configurations:
![](https://raw.githubusercontent.com/Cei1ing/AIClub2018_CV/master/Paper/VGGNet.JPG)

### Training:
1. Batch size 256.
2. Momentum 0.9, weight decay 0.0005, learning rate 0.01 (decrease by 10 third when accuracy stops improving).
3. Randomly initialize the shallow configuration, and use it as initialization for some part of deeper networks, rest with normal distribution with the zero mean and 0.01 variance.
-it is found possible to initialise the weights without pre-training by using the random initialisation procedure of Glorot & Bengio (2010).
4. ***Set training scale S, single-scale and multi-scale image statistics?***

### Testing:
1. Images are rescaled to a pre-defined scale Q not necessarily same as S.
2. ***FC layers are converted to convolutional layers, FCN densely applied to the whole image?***
3. Spatially average the resulted score map, augment by horizontal flipping, final score by average original and flipped soft-max.
4. Compared with multi-crop test:
   1. FCN does not need multiple crop.
   2. A large set of crops improves accuracy due to finrt sampling.
   3. ***Multi-crop is complementary to FCN?***
   4. Multi-crop takes more computation time.

### Results:
#### Single Scale Evaluation:
1. LRN is useless.
2. Additional non-linearity does help, but spatial context by conv.layers is also important.
3. A deep net with small filters outperforms a shallow net with larger filters.
4. Scale jittering at training time does help in spite of single scale at test time.
#### Multi-scale Evaluation:
1. Run a model over several rescaled versions of a test image (different values of Q), followed by averaging the resulting class.
2. Scale jittering at test time leads to better performance (as compared to evaluating the same model at a single scale).
#### Multi-crop Evaluation:
1. Averaging dense evaluation's soft-max and multi-crop's will get beeter result.
#### Fusion Evaluation:
1. Combine the outputs of several models by averaging their soft-max class posteriors.
2. Best result is achieved by combining just two models.
