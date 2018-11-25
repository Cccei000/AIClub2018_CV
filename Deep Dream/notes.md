# Deep Dream
## 1.实验思路：
核心思想是以CNN中隐含层的feature map的二范数为目标函数，对原输入进行梯度上升，从而增强网络所提取到的特征。这些特征显现在所得图像上，可以用于理解CNN每层究竟提取到了什么特征.  

## 2.代码实现：
实验采用的架构是Pytorch 0.4.1，此外还用到了numpy, scipy, PIL等package，所使用的预训练模型主要为ResNet-50. 

### 2.1.初期实验：
实验初期主要参考了[博客](https://www.jianshu.com/p/1ee5f5423850)中的两个技巧，其它部分由自己设计：
* 多尺度增强：先将图片缩小多次，然后从最小的图片开始增强，之后只获取增强的部分，将其尺度放大后加到更大的图片上进行增强，重复上述操作直到对原图进行增强.
* 抖动：在每次增强前保持通道不变，在画面空间上对图片进行大小随机的挪动，之后在进行增强，最后在挪动回原位.

然而最终得到的是烂掉的结果：
![](https://raw.githubusercontent.com/Cei1ing/AIClub2018_CV/master/Deep%20Dream/pic/failure_sky.png)  

### 2.2.进一步实验：
我从两个方面思考的得出了后续的改进方法：
1. Deep Dream和Adversarial Examples一样本质上都是梯度上升，所以后者的训练方法应当可以借鉴到前者中去.
2. 在最初还没有对实验进行调研时，自己随便写了段很粗糙的代码企图对噪声进行Deep Dream，结果怎么弄都看不到变化，最后发现是梯度实在太小了.

基于这两项观察，我采取了之前在研究Adversarial Examples时遇到的根据梯度自适应调整更新步长的方法，以保证每次更新的效果。具体而言，实际的步长是设定的步长除以梯度绝对值的平均数。这种方法之所以有效，我认为是因为它防止了微小而细碎的分支的产生，这些分支会在一次次增强中被进一步粉碎，从而导致结果烂掉.

## 3.实验结果：
以下展示不同设定下的部分实验结果，以及一些发现，更多结果请移步文pic/results：  

zoom_n = 6, zoom_ratio = 0.7, train = True, epochs = 30, max_jitter = 20, trg_layer = 3, lr = 0.01
![](https://raw.githubusercontent.com/Cei1ing/AIClub2018_CV/master/Deep%20Dream/pic/results/train/sky_07zoom6_3Res50.jpg)

zoom_n = 6, zoom_ratio = 0.7, train = False, epochs = 30, max_jitter = 20, trg_layer = 3, lr = 0.01
![](https://raw.githubusercontent.com/Cei1ing/AIClub2018_CV/master/Deep%20Dream/pic/results/eval/sky_07zoom6_3Res50.jpg)

zoom_n = 6, zoom_ratio = 0.7, train = False, epochs = 30, max_jitter = 20, trg_layer = 4, lr = 0.01
![](https://raw.githubusercontent.com/Cei1ing/AIClub2018_CV/master/Deep%20Dream/pic/results/eval/sky_07zoom6_4Res50.jpg)

zoom_n = 6, zoom_ratio = 0.7, train = True, epochs = 30, max_jitter = 20, trg_layer = 1, lr = 0.01
![](https://raw.githubusercontent.com/Cei1ing/AIClub2018_CV/master/Deep%20Dream/pic/results/train/city_07zoom6_1Res50.jpg)

zoom_n = 6, zoom_ratio = 0.7, train = True, epochs = 30, max_jitter = 20, trg_layer = 3, lr = 0.01
![](https://raw.githubusercontent.com/Cei1ing/AIClub2018_CV/master/Deep%20Dream/pic/results/train/city_07zoom6_3Res50.jpg)

zoom_n = 6, zoom_ratio = 0.7, train = False, epochs = 15, max_jitter = 20, trg_layer = 3, lr = 0.02
![](https://raw.githubusercontent.com/Cei1ing/AIClub2018_CV/master/Deep%20Dream/pic/img.jpg)
