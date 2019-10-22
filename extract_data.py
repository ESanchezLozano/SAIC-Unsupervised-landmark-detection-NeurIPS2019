from MTFAN import FAN, convertLayer, GeoDistill
import torch, numpy as np
from databases import SuperDB
from utils import *
from torch.utils.data import Dataset, DataLoader
import os, pickle

import argparse
parser = argparse.ArgumentParser(description='Extract data')
parser.add_argument('-f','--f', default='', type=str, metavar='PATH', help='folder')
parser.add_argument('-e','--e', default='', type=str, metavar='PATH', help='epoch')
parser.add_argument('-c','--core', default='checkpoint_fansoft/fan_109.pth', type=str, metavar='PATH', help='corenet')
parser.add_argument('-t','--t', default=16, type=int, metavar='PATH', help='tight')
parser.add_argument('-d','--db', default='MAFL-train', type=str, metavar='PATH', help='db')
parser.add_argument('--cuda', default='auto', type=str, help='cuda')
parser.add_argument('--data_path', default='', help='Path to the data')



def loadnet(npoints=10,path_to_model=None, path_to_core=None):
    net = FAN(1,n_points=npoints).to('cuda')
    checkpoint = torch.load(path_to_model)
    checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
    if path_to_core is not None:
        net_dict = net.state_dict()
        pretrained_dict = torch.load(path_to_core, map_location='cuda')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if pretrained_dict[k].shape == net_dict[k].shape}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict, strict=True)
        net.apply(convertLayer)
    net.load_state_dict(checkpoint)
    return net.to('cuda')

def getdata(loader, BOT, net):
    preds = []
    gths = []
    with torch.no_grad():
        for sample in loader:
            img = sample['Im']
            pts = sample['pts']
            _,out = BOT(net(img.cuda()))
            preds.append(out.cpu().detach())
            gths.append(pts)
    return np.concatenate(preds), np.concatenate(gths)

def extractdata(folder,epoch,path_to_core, db, tight, npoints, data_path):
    # outpickle
    outname = 'data_{}.pkl'.format(folder)
    # create model
    path_to_model = '{}/model_{}.fan.pth'.format(folder,epoch)
    #checkpoint = torch.load(path_to_model)['state_dict']
    net = loadnet(npoints,path_to_model, path_to_core)
    BOT = GeoDistill(sigma=0.5, temperature=0.1).to('cuda')
    # create data
    database = SuperDB(path=data_path,size=128,flip=False,angle=0.0,tight=tight or 64, db=db, affine=True)
    num_workers = 12 
    dbloader = DataLoader(database, batch_size=30, shuffle=False, num_workers=num_workers, pin_memory=False)
    # extract data        
    print('Extracting data from {:s}, with {:d} imgs'.format( db, len(database)))
    Ptr, Gtr = getdata(dbloader, BOT, net)
    # dump data
    data = pickle.load(open(outname,'rb')) if os.path.exists(outname) else {}
    if db not in data.keys():
        data[db] = {}    
    data[db][str(epoch)] = {'Ptr': Ptr, 'Gtr': Gtr}        
    pickle.dump(data, open(outname,'wb'))

def main():
    # input parameters    
    args = parser.parse_args()
    assert args.db in ['CelebA', 'AFLW-train', 'AFLW-test', 'MAFL-train', 'MAFL-test'], 'Please choose between CelebA, AFLW-train, AFLW-test, MAFL-train, MAFL-test'
    if args.cuda == 'auto':
        import GPUtil as GPU
        GPUs = GPU.getGPUs()
        idx = [GPUs[j].memoryUsed for j in range(len(GPUs))]
        print(idx)
        assert min(idx) < 11.0, 'All {} GPUs are in use'.format(len(GPUs))
        idx = idx.index(min(idx))
        print('Assigning CUDA_VISIBLE_DEVICES={}'.format(idx))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    folder, epoch = args.f, args.e
    path_to_core = args.core #'checkpoint_fansoft/fan_109.pth'
    db = args.db #'AFLW'
    tight = args.t 
    extractdata(folder,epoch,path_to_core, db, tight, npoints=10, data_path=args.data_path)


if __name__ == '__main__':
    main()