import argparse
import os

class Options():

    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Unsupervised Landmark Discovery through unsupervised adaptation (NeurIPS19)')
        self.initialize()
        self.args = self.parse_args()
        self.write_args()

    def initialize(self):
        self._parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint (default: none)')
        self._parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
        self._parser.add_argument('--file', default='model_',help='Store model')
        self._parser.add_argument('-o','--option', type=str, default='incremental', help='Use incremental/finetune/scratch')
        self._parser.add_argument('-f','--folder', default='.', help='Folder to save intermediate models')
        self._parser.add_argument('--bSize', default=48, help='Batch Size')
        self._parser.add_argument('--s',default=0.5,help='Sigma for heatmaps')
        self._parser.add_argument('--t',default=0.1,help='Temperature of Softargmax')
        self._parser.add_argument('--gc', default=1, help='Use gradient clipping')
        self._parser.add_argument('--tight', default=70, help='Tight')
        self._parser.add_argument('--npts', default=10, help='Number of points')
        self._parser.add_argument('--size', default=128, type=int, help='Size of images')
        self._parser.add_argument('--num_workers', default=12, help='Number of workers')
        self._parser.add_argument('--visdom', default=True, help='Window for Visdom')
        self._parser.add_argument('--data_path', default='', help='Path to the data')
        self._parser.add_argument('--port', default=9001, help='visdom port')
        self._parser.add_argument('--db', default='CelebA', help='db')
        self._parser.add_argument('--checkpoint', default='checkpoint_fansoft/fan_109.pth')
        self._parser.add_argument('-lf','--lr_fan', default=0.001, type=float, metavar='PATH', help='learning rate fan')
        self._parser.add_argument('-lg','--lr_gan', default=0.001, type=float, metavar='PATH', help='learning rate gan')
        self._parser.add_argument('-sz','--step_size', default=30, type=int, help='Step size for scheduler')
        self._parser.add_argument('-g', '--gamma', default=0.1, type=float, help='Gamma for scheduler')
        self._parser.add_argument('-a', '--angle', default=15.0, type=float, help='rotation angle')
        self._parser.add_argument('--flip', default=False, help='Use flip or not')
        self._parser.add_argument('--cuda', default='auto', type=str, help='cuda')

    def parse_args(self):
        self.args = self._parser.parse_args()
        if self.args.folder == '.':
            experimentname = sorted([l for l in os.listdir(os.getcwd()) if os.path.isdir(l) and l.find('Exp') > -1])
            self.args.folder = 'Exp_{:d}'.format(len(experimentname))
        self.args.visdom = self.args.folder if eval(str(self.args.visdom)) else None
        print(self.args.folder)
        return self.args

    def write_args(self):
        if not os.path.exists('./' + self.args.folder):
            os.mkdir('./' + self.args.folder)
        if not os.path.exists(os.path.join(self.args.folder, 'code')):
            os.mkdir(os.path.join(self.args.folder,'code'))
        with open(self.args.folder + '/args_' + self.args.file[0:-8] + '.txt','w') as f:
            print(' '.join(['--{:s} {} '.format(k, self.args.__getattribute__(k)) for k in list(self.args.__dict__.keys())]) + '\n',file=f)
        



