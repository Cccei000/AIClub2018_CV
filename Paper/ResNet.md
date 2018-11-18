### Background:
* Degradation: with the network depth increasing, accuracy gets saturated and then degrades rapidly.
* Current solvers (multiple nonlinear layers) adding identity mapping layers are unable to find solutions.

### Deep Residual Learning:
* Instead of the original mapping H(x), learn H(x) - x, easier and faster for learning.
* Applying shortcuts.  
-***how is this implemented when dimensions increase?***

### Basicblock Architecture:
* For each residual function F, use a stack of 2 layers.
* The two layers are 3x3x64 and 3x3x64 convolutions.

### Bottleneck Architecture：
* For each residual function F, use a stack of 3 layers instead of 2. 
* The three layers are 1×1x64, 3×3x64, and 1×1x64 convolutions.
* 1×1 layers are responsible for reducing and then increasing dimensions, leaving the 3×3 layer smaller input/output dimensions.
