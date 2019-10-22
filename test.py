import pickle, torch, numpy as np, os
import argparse
parser = argparse.ArgumentParser(description='Extract data')
parser.add_argument('-f','--f', default='', type=str, metavar='PATH', help='folder')
parser.add_argument('-d','--db', default='MAFL', type=str, metavar='PATH', help='db')
parser.add_argument('-m', '--mode', default='fwd', type=str, help='Forward')
parser.add_argument('-p', '--npoints', default=10, type=int, help='number of points')
parser.add_argument('-e', '--epoch', default=0, type=int, help='Epoch to test')
parser.add_argument('-r', '--reg', default=0.00001, type=float, help='Regularization factor')


def main():
    # input parameters
    args = parser.parse_args()
    npoints = args.npoints
    folder = args.f
    folder = folder.replace('/','')
    assert os.path.exists(f'data_{folder}.pkl'), 'Data not extracted yet'
    data = pickle.load(open('data_{}.pkl'.format(folder),'rb'))
    db = args.db  
    fwd = args.mode == 'fwd'
    output_file = 'allerrors_{}_{}.pkl'.format(folder, 'fwd' if fwd else 'bwd')
    if fwd:
        print('Forward training --------- ')
    else:
        print('Backward training --------- ')
    all_errors = {} if not os.path.exists(output_file) else pickle.load(open(output_file,'rb'))
    reg_factor = args.reg
    e = str(args.epoch)
    print('Doing epoch {}'.format(e))
    all_errors[e] = []
    dbtrain = db + '-train'
    dbtest = db + '-test'
    Ytr = data[dbtrain][e]['Gtr']
    Ytr = Ytr.reshape(Ytr.shape[0],-1)/4
    Xtr = data[dbtrain][e]['Ptr']
    Xtr = Xtr.reshape(Xtr.shape[0],-1)
    Ytest = data[dbtest][e]['Gtr']
    Ytest = Ytest.reshape(Ytest.shape[0],-1)/4
    Xtest = data[dbtest][e]['Ptr']
    Xtest = Xtest.reshape(Xtest.shape[0],-1)
    all_errors[e].append(compute_errors(Xtr,Ytr,Xtest,Ytest,reg_factor,True) if fwd else compute_errors(Ytr,Xtr,Ytest,Xtest,reg_factor,False))
    pickle.dump(all_errors, open(output_file,'wb'))
    print(all_errors)

def compute_errors(Xtr,Ytr,Xtest,Ytest,reg_factor,fwd=True):
    npts = 10
    n = [1,5,10,100,500,1000,5000,Xtr.shape[0]]
    nrepeats = 10
    all_errors = np.zeros((len(n),nrepeats))
    for tmp_idx in range(0,len(n)):
        for j in range(0,nrepeats):
            idx = np.random.permutation((range(0,Xtr.shape[0])))[0:n[tmp_idx]+1]
            R, X0, Y0 = train_regressor(Xtr[idx,:], Ytr[idx,:], reg_factor)
            err = np.zeros((Xtest.shape[0]))
            for i in range(0,Xtest.shape[0]):
                x = Xtest[i,:]
                y = Ytest[i,:]
                if fwd:
                    x = fit_regressor(R,x,X0,Y0)
                    err[i] = NMSE( y.reshape(-1,2), x)
                else:
                    iod = compute_iod(x.reshape(-1,2))
                    x = fit_regressor(R,x,X0,Y0)
                    y = y.reshape(-1,2)
                    err[i] = np.sum(np.sqrt(np.sum((x-y)**2,1)))/(iod*npts)
            all_errors[tmp_idx,j] = np.mean(err)
    return all_errors        

def NMSE(landmarks_gt, landmarks_regressed):
    if len(landmarks_gt.shape) == 2:
        landmarks_gt = landmarks_gt.reshape(1,-1,2)
    if len(landmarks_regressed.shape) == 2:
        landmarks_regressed = landmarks_regressed.reshape(1,-1,2)
    eyes = landmarks_gt[:, :2, :] if landmarks_gt.shape[1] == 5 else landmarks_gt[:,[36,45],:]
    occular_distances = np.sqrt(np.sum((eyes[:, 0, :] - eyes[:, 1, :])**2, axis=-1))
    distances = np.sqrt(np.sum((landmarks_gt - landmarks_regressed)**2, axis=-1))
    mean_error = np.mean(distances / occular_distances[:, None])
    return mean_error

def train_regressor(X,Y,l,size=64):
    center = size/2
    Xtmp = X/center - 0.5
    X0 = Xtmp.mean(axis=0, keepdims=True)
    Xtmp = Xtmp - np.ones((Xtmp.shape[0],1)) @ X0.reshape(1,-1)
    C = Xtmp.transpose() @ Xtmp
    Ytmp = Y/center - 0.5
    Y0 = Ytmp.mean(axis=0, keepdims=True)
    Ytmp = Ytmp - np.ones((Ytmp.shape[0],1)) @ Y0.reshape(1,-1)
    R = ( Ytmp.transpose() @ Xtmp ) @ np.linalg.inv( C + l*(C.max()+1e-12)*np.eye(Xtmp.shape[1])) 
    return R, X0, Y0

def fit_regressor(R,x,X0,Y0, size=64):
    center = size/2
    x = (R @ (x/center - 0.5 - X0).transpose()).reshape(-1,2) + Y0.reshape(-1,2)
    x = (x + 0.5)*center
    return x

def compute_iod(y):
    if y.shape[0] == 5:
        iod = np.sqrt( (y[0,0] - y[1,0])**2 + (y[0,1] - y[1,1])**2 )
    else:
        iod = np.sqrt( (y[36,0] - y[45,0])**2 + (y[36,1] - y[45,1])**2 )
    return iod


if __name__ == '__main__':
    main()