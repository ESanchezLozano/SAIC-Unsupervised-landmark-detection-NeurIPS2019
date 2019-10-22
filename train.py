from __future__ import print_function, division
import glob, os, sys, pickle, torch, cv2, time, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from shutil import copy2
# mystuff
from model import model as mymodel
from databases import SuperDB
from utils import *
from Train_options import Options

def main():
    # parse args
    global args
    args = Options().args
    
    # copy all files from experiment
    cwd = os.getcwd()
    for ff in glob.glob("*.py"):
        copy2(os.path.join(cwd,ff), os.path.join(args.folder,'code'))

    # initialise seeds
    torch.manual_seed(1000)
    torch.cuda.manual_seed(1000)
    np.random.seed(1000)

    # choose cuda
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

    # parameters
    sigma = float(args.s)
    temperature = float(args.t)
    gradclip = int(args.gc)
    npts = int(args.npts)
    bSize = int(args.bSize)
    angle = float(args.angle)
    flip = eval(str(args.flip))
    tight = int(args.tight)

    model = mymodel(sigma=sigma,temperature=temperature, gradclip=gradclip, npts=npts, option=args.option, size=args.size, path_to_check=args.checkpoint)
    
    plotkeys = ['input','target','generated']
    losskeys = list(model.loss.keys())
    
    # define plotters
    global plotter
    if not args.visdom:
        print('No Visdom')
        plotter = None
    else:
        from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomSaver, VisdomTextLogger
        experimentsName = str(args.visdom)
        plotter = dict.fromkeys(['images','losses'])
        plotter['images'] = dict( [ (key, VisdomLogger("images", port=int(args.port), env=experimentsName, opts={'title' : key})) for key in plotkeys ])
        plotter['losses'] = dict( [ (key, VisdomPlotLogger("line", port=int(args.port), env=experimentsName,opts={'title': key, 'xlabel' : 'Iteration', 'ylabel' : 'Loss'})) for key in losskeys]  )
        
    # prepare average meters
    global meters, l_iteration
    meterskey = ['batch_time', 'data_time'] 
    meters = dict([(key,AverageMeter()) for key in meterskey])
    meters['losses'] = dict([(key,AverageMeter()) for key in losskeys])
    l_iteration = float(0.0)
    
    # plot number of parameters
    params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.GEN.parameters())])
    print('GEN # trainable parameters: {}'.format(params))
    params = sum([p.numel() for p in filter(lambda p: p.requires_grad, model.FAN.parameters())])
    print('FAN # trainable parameters: {}'.format(params))

    
    
    # define data
    video_dataset = SuperDB(path=args.data_path,sigma=sigma,size=args.size,flip=flip,angle=angle,tight=tight, db=args.db)
    videoloader = DataLoader(video_dataset, batch_size=bSize, shuffle=True, num_workers=int(args.num_workers), pin_memory=True)
    print('Number of workers is {:d}, and bSize is {:d}'.format(int(args.num_workers),bSize))
       
    # define optimizers
    lr_fan = args.lr_fan
    lr_gan = args.lr_gan
    print('Using learning rate {} for FAN, and {} for GAN'.format(lr_fan,lr_gan))
    optimizerFAN = torch.optim.Adam(model.FAN.parameters(), lr=lr_fan, betas=(0, 0.9), weight_decay=5*1e-4)
    schedulerFAN = torch.optim.lr_scheduler.StepLR(optimizerFAN, step_size=args.step_size, gamma=args.gamma)
    optimizerGEN = torch.optim.Adam(model.GEN.parameters(), lr=lr_gan, betas=(0, 0.9), weight_decay=5*1e-4)
    schedulerGEN = torch.optim.lr_scheduler.StepLR(optimizerGEN, step_size=args.step_size, gamma=args.gamma)
    myoptimizers = {'FAN' : optimizerFAN, 'GEN' : optimizerGEN}

    # path to save models and images
    path_to_model = os.path.join(args.folder,args.file)

    # train
    for epoch in range(0,80):
        schedulerFAN.step()
        schedulerGEN.step()
        train_epoch(videoloader, model, myoptimizers, epoch, bSize)
        model._save(path_to_model,epoch)


def train_epoch(dataloader, model, myoptimizers, epoch, bSize):
    
    itervideo = iter(dataloader)
    global l_iteration
    log_epoch = {}
    end = time.time()
    for i in range(0,2500):
    
        
        # - get data
        all_data = next(itervideo,None) 
        if all_data is None:
            itervideo = iter(dataloader)
            all_data = next(itervideo, None)
        elif all_data['Im'].shape[0] < bSize:
            itervideo = iter(dataloader)
            all_data = next(itervideo, None)
        
        # - set batch
        model._set_batch(all_data)
        
        # - forward
        output = model.forward()
        
        # - update parameters
        myoptimizers['GEN'].step()
        myoptimizers['FAN'].step()
                
        meters['losses']['rec'].update(model.loss['rec'].item(), bSize)
        l_iteration = l_iteration + 1

        
        if i % 100 == 0:                
            # - plot some images
            allimgs = None
            for (ii,imtmp) in enumerate(all_data['Im'].to('cpu').detach()):
                improc = (255*imtmp.permute(1,2,0).numpy()).astype(np.uint8).copy()
                x = 4*output['Points'][ii]
                for m in range(0,x.shape[0]):
                    cv2.circle(improc, (int(x[m,0]), int(x[m,1])), 3, (255,0,0),-1)
                if allimgs is None:
                    allimgs = np.expand_dims(improc,axis=0)
                else:
                    allimgs = np.concatenate((allimgs, np.expand_dims(improc,axis=0)))

            if plotter is not None:
                plotter['images']['input'].log(torch.from_numpy(allimgs).permute(0,3,1,2))
                plotter['images']['target'].log(all_data['ImP'].data)            
                plotter['images']['generated'].log(output['Reconstructed'].cpu().data)
                plotter['losses']['rec'].log( l_iteration, model.loss['rec'].item() )
            
            save = torch.nn.functional.interpolate(torch.from_numpy(allimgs/255.0).permute(0,3,1,2),scale_factor=0.25)
            save_image(save, args.folder + '/image_{}_{}.png'.format(epoch,i))
                    
               
        log_epoch[i] = model.loss       
        meters['batch_time'].update(time.time()-end)
        end = time.time()
        if i % args.print_freq == 0:
            mystr = 'Epoch [{}][{}/{}] '.format(epoch, i, len(dataloader))
            mystr += 'Time {:.2f} ({:.2f}) '.format(meters['data_time'].val , meters['data_time'].avg )
            mystr += ' '.join(['Loss: {:s} {:.3f} ({:.3f}) '.format(k, meters['losses'][k].val , meters['losses'][k].avg ) for k in meters['losses'].keys()])
            print( mystr )
            with open(args.folder + '/args_' + args.file[0:-8] + '.txt','a') as f: 
                print( mystr , file=f)

    with open(args.folder + '/args_' + args.file[0:-8] + '_' + str(epoch) + '.pkl','wb') as f:
        pickle.dump(log_epoch,f)  

if __name__ == '__main__':
    main()


