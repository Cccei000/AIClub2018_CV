import numpy as np
import scipy.ndimage as nd
import torch as t
import utils


device = t.device('cuda' if t.cuda.is_available() else 'cpu')


def make_dreams(img,model,epochs,max_jitter,trg_layer,lr):
    img = np.array([img])
    
    for i in range(epochs):
        jitterH,jitterW = np.random.randint(-max_jitter,max_jitter+1,2)
        img = np.roll(img,jitterH,2)
        img = np.roll(img,jitterW,3)
        img = t.tensor(img).to(device)
        img.requires_grad = True
    
        #optimizer = t.optim.SGD([img],momentum = momentum,lr = lr)
        model.zero_grad()
        
        activation = model.forward(img,trg_layer)
        objective = t.sum(activation**2)
        objective.backward()
        #activation.backward(activation.data)
        #optimizer.step()
        
        ratio = np.abs(img.grad.data.cpu().numpy()).mean()
        img.data.add_(img.grad.data*(lr/ratio))
                
        img.grad.data.zero_()
        img = img.detach().cpu().numpy()
        img = np.roll(img,-jitterW,3)
        img = np.roll(img,-jitterH,2)
       
    img = img[0,:,:,:]
    
    return img    
    
    
def dreamer(model,input_img,zoom_n,zoom_ratio,train,
            epochs = 20,max_jitter = 32,trg_layer = 3,lr = 0.02):
    
    model = model.to(device)
    if train == False:
        model.eval()
    elif train == True:
        model.train()
    for param in model.parameters():
        param.requires_grad = False
    
    zooms = [input_img] # img shape (channel,height,width)
    for i in range(zoom_n):
        zooms.append(nd.zoom(zooms[-1],(1,zoom_ratio,zoom_ratio),order = 1))
        
    augment = np.zeros_like(zooms[-1])
    
    for i in range(zoom_n+1):
        if i > 0:
            dstH,dstW = zooms[zoom_n-i].shape[1:3]
            srcH,srcW = augment.shape[1:3]
            augment = nd.zoom(augment,(1,dstH/srcH,dstW/srcW),order = 1)
        
        dream_src = zooms[zoom_n-i]+augment
        dream = make_dreams(dream_src,model,epochs,max_jitter,trg_layer,lr)
        augment = dream-zooms[zoom_n-i]
        
        print("Step {} out of {} finished.".format(i+1,zoom_n+1))
    
    utils.showImg(augment+input_img)