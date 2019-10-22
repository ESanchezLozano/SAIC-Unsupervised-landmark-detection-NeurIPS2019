import os, sys, time, torch, math, numpy as np, cv2, collections
import torch.nn as nn
import torch.nn.functional as F

################################### - classes 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class HeatMap(torch.nn.Module):
    """Defines a differentiable Gaussian heatmap"""
    def __init__(self, out_res, sigma=0.5):
        super(HeatMap, self).__init__()
        self.sigma = sigma
        print('The size of heatmap is {:f}'.format(self.sigma))
        y,x = torch.meshgrid([torch.arange(0,out_res).float(), torch.arange(0,out_res).float()])
        self.x = x
        self.y = y
        self.out_res = out_res
    
    def forward(self, pts):
        bSize, nPts = pts.size(0), pts.size(1)
        x = self.x.repeat(bSize,nPts,1,1)
        y = self.y.repeat(bSize,nPts,1,1)
        xscore = torch.unsqueeze(torch.unsqueeze(pts[:,:,0], 2),3)
        yscore = torch.unsqueeze(torch.unsqueeze(pts[:,:,1], 2),3)
        xscore = xscore - x.to(xscore.device)
        yscore = yscore - y.to(yscore.device)
        hms = -(xscore**2 + yscore**2)
        hms = torch.exp(hms/self.sigma)
        return hms
    
class SoftArgmax2D(torch.nn.Module):
    """ Implementation of a 2d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations."""
    def __init__(self, base_index=0, step_size=1, softmax_temp=1.0):
        super(SoftArgmax2D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=2)
        self.softmax_temp = softmax_temp

    def _softmax_2d(self, x, temp):
        B, C, W, H = x.size()
        x_flat = x.view((B, C, W*H)) / temp
        x_softmax = self.softmax(x_flat)
        return x_softmax.view((B, C, W, H))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        smax = self._softmax_2d(x, self.softmax_temp)# * windows
        smax = smax / torch.sum(smax.view(batch_size, channels, -1), dim=2).view(batch_size,channels,1,1)
        # compute x index (sum over y axis, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).float().cuda()
        x_coords = torch.sum(torch.sum(smax, 2) * x_indices, 2)
        # compute y index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).float().cuda()
        y_coords = torch.sum(torch.sum(smax, 3) * y_indices, 2)
        return torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)


class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.LossOutput = collections.namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        self.vgg_layers = vgg_model.features if hasattr(vgg_model,'features') else vgg_model.module.features #### to allow use in DataParallel
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return self.LossOutput(**output)


################################ functions

def process_image(image,points,angle=0, flip=False, sigma=1,size=128, tight=16, hmsize=64):
    output = dict.fromkeys(['image','points','M'])
    if angle > 0:
        tmp_angle = np.clip(np.random.randn(1) * angle, -40.0, 40.0)
        image,points,M = affine_trans(image,points, tmp_angle)
        output['M'] = M
        tight = int(tight + 4*np.random.randn())
    image, points = crop( image , points, size, tight )
    if flip:
        image = cv2.flip(image, 1)
            
    image = image/255.0
    image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
    output['image'] = image.type_as(torch.FloatTensor())
    output['points'] = np.floor(points)

    return output


def crop( image, landmarks , size, tight=8):
        delta_x = np.max(landmarks[:,0]) - np.min(landmarks[:,0])
        delta_y = np.max(landmarks[:,1]) - np.min(landmarks[:,1])
        delta = 0.5*(delta_x + delta_y)
        if delta < 20:
            tight_aux = 8
        else:
            tight_aux = int(tight * delta/100)
        pts_ = landmarks.copy()
        w = image.shape[1]
        h = image.shape[0]
        min_x = int(np.maximum( np.round( np.min(landmarks[:,0]) ) - tight_aux , 0 ))
        min_y = int(np.maximum( np.round( np.min(landmarks[:,1]) ) - tight_aux , 0 ))
        max_x = int(np.minimum( np.round( np.max(landmarks[:,0]) ) + tight_aux , w-1 ))
        max_y = int(np.minimum( np.round( np.max(landmarks[:,1]) ) + tight_aux , h-1 ))
        image = image[min_y:max_y,min_x:max_x,:]
        pts_[:,0] = pts_[:,0] - min_x
        pts_[:,1] = pts_[:,1] - min_y
        sw = size/image.shape[1]
        sh = size/image.shape[0]
        im = cv2.resize(image, dsize=(size,size),
                        interpolation=cv2.INTER_LINEAR)
        
        pts_[:,0] = pts_[:,0]*sw
        pts_[:,1] = pts_[:,1]*sh
        return im, pts_



def affine_trans(image,landmarks,angle=None,size=None):
    if angle is None:
        angle = 30*torch.randn(1)
       
    
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    dst = cv2.warpAffine(image, M, (nW,nH),borderMode=cv2.BORDER_REPLICATE)
    #print(landmarks.shape)
    new_landmarks = np.concatenate((landmarks,np.ones((landmarks.shape[0],1))),axis=1)
    if size is not None:
        sw = size/nW
        sh = size/nH
        dst = cv2.resize(dst, dsize=(size,size),
                        interpolation=cv2.INTER_LINEAR)
        M = [[sw,0],[0,sh]] @ M
    
    new_landmarks = new_landmarks.dot(M.transpose())
    return dst, new_landmarks, M



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

############################# - Visualisation utils

def savetorchimg(name,img):
    cv2.imwrite(name, cv2.cvtColor((255*img.permute(1,2,0).numpy()).astype(np.uint8) , cv2.COLOR_RGB2BGR))

def savetorchimgandpts(name,img,x,y=None):
    improc = (255*img.permute(1,2,0).numpy()).astype(np.uint8).copy()
    for m in range(0,x.shape[0]):
        cv2.circle(improc, (int(x[m,0]), int(x[m,1])), 2, (255,0,0),-1)
    if y is not None:
        for m in range(0,y.shape[0]):
            cv2.circle(improc, (int(y[m,0]), int(y[m,1])), 2, (0,255,0),-1)
    cv2.imwrite(name, cv2.cvtColor( improc , cv2.COLOR_RGB2BGR))


def saveheatmap(name,img):
    improc = cv2.applyColorMap( (255*img.permute(1,2,0).numpy()).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(name, cv2.cvtColor( improc , cv2.COLOR_RGB2BGR))


def savetorchimgandptsv2(name,img,x,thick=2,mSize=4): # to use different colours
    improc = (255*img.permute(1,2,0).numpy()).astype(np.uint8).copy()
    cv2.drawMarker(improc, (int(x[0,0]), int(x[0,1])), (255,0,0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[1,0]), int(x[1,1])), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[2,0]), int(x[2,1])), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[3,0]), int(x[3,1])), (0,0,0), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[4,0]), int(x[4,1])), (255,255,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[5,0]), int(x[5,1])), (255,255,0), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[6,0]), int(x[6,1])), (255,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[7,0]), int(x[7,1])), (0,255,255), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[8,0]), int(x[8,1])), (255,128,0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=mSize, thickness=thick)
    cv2.drawMarker(improc, (int(x[9,0]), int(x[9,1])), (0,0,128), markerType=cv2.MARKER_CROSS, markerSize=mSize, thickness=thick)       
    cv2.imwrite(name, cv2.cvtColor( improc , cv2.COLOR_RGB2BGR))

