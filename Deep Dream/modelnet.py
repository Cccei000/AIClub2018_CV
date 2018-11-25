import torchvision.models.resnet as resnet
import torchvision.models.inception as inception
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class ResNet(resnet.ResNet):
    def forward(self,x,trg_layer):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        trg_layers = [self.layer1,self.layer2,self.layer3,self.layer4]
        
        for i in range(trg_layer):
            x = trg_layers[i](x)
            
        return x
    
    
class Inception(inception.Inception3):
    def forward(self,x,end_layer):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        if end_layer == 1:
            return x
        
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        if end_layer == 2:
            return x
        
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        if end_layer == 3:
            return x
        
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        if end_layer == 4:
            return x
        
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        return x

    
def res18(pretrained = False):
    model = ResNet(resnet.BasicBlock,[2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']))
    return model     
 
    
def res50(pretrained = False):
    model = ResNet(resnet.Bottleneck,[3,4,6,3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
    return model


def inception3(pretrained = False):
    model = Inception()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(inception.model_urls['inception_v3_google']))

    return model