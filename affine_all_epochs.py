from MTFAN import FAN, convertLayer, GeoDistill
import torch, numpy as np, glob, re
from utils import *
from torch.utils.data import Dataset, DataLoader
import os, pickle
from databases import SuperDB           
import argparse
parser = argparse.ArgumentParser(description='Extract data')
parser.add_argument('-f','--f', default='', type=str, metavar='PATH', help='folder')
parser.add_argument('-c','--core', default='', type=str, metavar='PATH', help='corenet')
parser.add_argument('-d','--db', default='MAFL-test', type=str, metavar='PATH', help='db')
parser.add_argument('-n', '--npts', default=10, type=int, metavar='PATH', help='number of points')
parser.add_argument('-t', '--tight', default=16, type=int, help='tight')
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

def evalaffine(facenet, db, npts=10):
    errors = np.zeros((len(db),npts))
    trainloader = DataLoader(db, batch_size=30, shuffle=False, num_workers=12, pin_memory=False)
    i = 0
    BOT = GeoDistill(sigma=0.5, temperature=0.1).to('cuda')
    for j, sample in enumerate(trainloader):
        a,b,c = sample['Im'], sample['ImP'], sample['M']
        _,preda = BOT(facenet(a.cuda()))
        _,predb = BOT(facenet(b.cuda()))
        pred_b = []
        for m in range(preda.shape[0]):
            pred_b.append(torch.cat((4*preda[m].cpu(), torch.ones(npts,1)),dim=1) @ c[m].permute(1,0))
            errors[i,:] = np.sqrt(np.sum((pred_b[m].detach().numpy() - 4*predb[m].detach().cpu().numpy())**2, axis=-1))
            i = i + 1
    return errors


def main():

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

    folder = args.f
    path_to_core = args.core if len(args.core) > 0 else None#'checkpoint_fansoft/fan_109.pth'
    if path_to_core is not None:
        if path_to_core.find('/') == -1:
            path_to_core = os.path.join('checkpoint_fansoft', path_to_core)

    folder = folder.replace('/','')
    db = args.db 
    tight = args.tight
    output_file = 'affine_{}.pkl'.format(folder)
    
    db = SuperDB(path=args.data_path, size=128,flip=False,angle=15.0,tight=tight or 64, db=db, affine=True)
    files = list(map(lambda x: x.split('/')[1], glob.glob('{}/model_*.fan.pth'.format(folder))))
    epochs = sorted([int(j) for j in [''.join(re.findall(r'\d+', k)) for k in files] if len(j) > 0])
    
    all_errors = pickle.load(open(output_file,'rb')) if os.path.exists(output_file) else {}
    epochs = [k for k in epochs if k not in all_errors.keys()]
    for e in epochs:
        print('Affine experiment epoch {:d} for folder {:s} out of {:d} epochs'.format(e,folder,epochs[-1]))
        path_to_model = '{}/model_{}.fan.pth'.format(folder,str(e))
        net = loadnet(npoints=args.npts,path_to_model=path_to_model, path_to_core=path_to_core)
        errors = evalaffine(net,db,npts=args.npts)
        print(np.mean(errors,0))
        all_errors[e] = np.mean(errors,0)
        pickle.dump(all_errors,open(output_file,'wb'))


if __name__ == '__main__':
    main()