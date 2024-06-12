import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import mydataset
import argparse
import time
from model import get_model
import mindspore.ops as ops
# import torch
import numpy as np
import myloss
# from torch import nn
# from torch.optim import Adam, SGD
# from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import mindspore as ms
from mindspore import nn, Tensor
import mindspore.numpy as mnp
import copy 
import matplotlib.pyplot as plt
from utils import saveImg,saveSingleImg
from sklearn.cluster import KMeans
import scipy.io as scio
from evaluation import clustering_metric
from constructGraph import getMvKNNGraph
isnan = ops.IsNan()
def train_1(loader, dataset, model,all_graph,all_encX, loss_model, optimizer, sche, estimator, epoch,logger):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    rec_losses = AverageMeter()
    exi_losses = AverageMeter()
    all_ori = [v_data for v_data in dataset.mv_data]
    # all_ori = [torch.tensor([])]*dataset.view_num
    all_out = [np.array([])]*dataset.view_num
    all_ind = np.array([])
    
    # all_newX = [torch.tensor([])]*dataset.view_num
    all_label = dataset.cur_labels
    mse = nn.MSELoss()
    # model.train()
    model.recover=True
    model.set_train(True)
    end = time.time()
    for i, (*data, label, inc_V_ind) in enumerate(loader):
        data,Cdata = data[:len(data)//2],data[len(data)//2:]
        data_time.update(time.time() - end)
        # label = label.to('cuda:0')
        
        def forward_fn(data,inc_V_ind,all_encX):
            # inc_V_ind = mnp.ones_like(inc_V_ind_)
            encX,decX,x_bar,_,emb_in,emb_out = model(copy.deepcopy(data),mask=copy.deepcopy(inc_V_ind))
            # x_bar = model(copy.deepcopy(data),inc_V_ind)
            all_encX[i*args.batch_size:i*args.batch_size+label.shape[0],:,:] = encX
            graph_loss = 0
            # print(x_bar[0])
            # print(data)
            mse_loss = loss_model.weighted_wmse_loss(x_bar,data,inc_V_ind)
            if epoch >0: 
                graph_loss = loss_model.graph_loss(all_graph[:,i*args.batch_size:i*args.batch_size+label.shape[0]],encX.transpose(1,0,2),all_encX.transpose(1,0,2))
            loss = mse_loss*1+graph_loss*args.beta
            # loss = mse_loss
            # print('loss',mse_loss,graph_loss)
            return loss,all_encX,*x_bar
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        
        (loss,all_encX,*x_bar), grads = grad_fn(data,inc_V_ind,Tensor(all_encX))

        optimizer(grads)

        losses.update(loss.numpy())
        batch_time.update(time.time()- end)
        end = time.time()
        
        
        all_out = [np.concatenate((all_out[j],v_data.numpy()),0) if len(all_out[j])>0 else v_data.numpy() for j,v_data in enumerate(copy.deepcopy(x_bar))]
        all_ind = np.concatenate((all_ind,inc_V_ind.numpy()),0) if len(all_ind)>0 else inc_V_ind.numpy()
 
        # encX=copy.deepcopy(encX.asnumpy())
        

    all_newX = copy.deepcopy(all_ori)
    for v,v_data in enumerate(all_out):
        
        all_newX[v][(1-all_ind[:,v])==1] = copy.deepcopy(v_data[(1-all_ind[:,v])==1])
        

    # results = evaluate(all_com.numpy(),all_label,estimator,dataset.classes_num,epoch,logger)
    if epoch%5==0 and epoch>0:

        rec_losses.update(np.mean([mse(Tensor(all_out[v][all_ind[:,v]==0]),Tensor(v_data[all_ind[:,v]==0])).numpy() for v,v_data in enumerate(all_ori)]))
        exi_losses.update(np.mean([mse(Tensor(all_out[v][all_ind[:,v]==1]),Tensor(v_data[all_ind[:,v]==1])).numpy() for v,v_data in enumerate(all_ori)]))
        print('rec_loss:{:.4f}  exi_loss:{:.4f}'.format(rec_losses.vals[-1],exi_losses.vals[-1]))

    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model,all_encX,all_newX


def train_2(loader, dataset, model,all_graph,all_encX,all_newX, loss_model, optimizer, sche, estimator, epoch,logger,fold_idx):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    rec_losses = AverageMeter()
    exi_losses = AverageMeter()
    all_ori = [v_data for v_data in dataset.mv_data]
    # all_ori = [torch.tensor([])]*dataset.view_num
    all_out = [np.array([])]*dataset.view_num
    all_ind = np.array([])
    all_com = np.array([])
    # all_encX_copy = copy.deepcopy(all_encX) 
    all_label = dataset.cur_labels
    mse = nn.MSELoss()
    model.set_train(True)
    model.recover=False
    end = time.time()
    for i, (*data, label, inc_V_ind) in enumerate(loader):
        data,Cdata = data[:len(data)//2],data[len(data)//2:]
        data_time.update(time.time() - end)
        # data = [v_newX[i*args.batch_size:i*args.batch_size+label.shape[0]] for v_newX in all_newX]
        data = [Tensor(v_newX[i*args.batch_size:i*args.batch_size+label.shape[0]]) for v_newX in all_newX]
        # label = label.to('cuda:0')

        def forward_fn(data,inc_V_ind,all_encX):
            encX,decX,x_bar,H,emb_in,emb_out = model(copy.deepcopy(data),mask=ops.ones_like(inc_V_ind))
            # x_bar = model(copy.deepcopy(data),inc_V_ind)
            all_encX[i*args.batch_size:i*args.batch_size+label.shape[0],:,:] = encX
            graph_loss = 0
            
            mse_loss = loss_model.weighted_wmse_loss(x_bar,data,ops.ones_like(inc_V_ind))
            graph_loss = loss_model.graph_loss(all_graph[:,i*args.batch_size:i*args.batch_size+label.shape[0]],encX.transpose(1,0,2),all_encX.transpose(1,0,2))
            loss = mse_loss*1+graph_loss*args.beta
            return loss,H,all_encX,*x_bar
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        (loss,H,all_encX,*x_bar), grads = grad_fn(data,inc_V_ind,Tensor(all_encX))
        optimizer(grads)
        # print(model.classifier.parameters().grad)
        losses.update(loss.numpy())
        batch_time.update(time.time()- end)
        end = time.time()

        all_out = [np.concatenate((all_out[j],v_data.numpy()),0) if len(all_out[j])>0 else v_data.numpy() for j,v_data in enumerate(copy.deepcopy(x_bar))]
        all_ind = np.concatenate((all_ind,inc_V_ind.numpy()),0) if len(all_ind)>0 else inc_V_ind.numpy()
        all_com = np.concatenate((all_com,H.numpy())) if len(all_com)>0 else H.numpy()
        
    results = {'ACC': 0., 'AMI': 0., 'NMI': 0., 'ARI': 0., 'PUR': 0.}
    # results = evaluate(all_com.numpy(),all_label,estimator,dataset.classes_num,epoch,logger)
    if epoch%5==0 and epoch>0:
        results = evaluate(all_com,all_label,estimator,dataset.classes_num,epoch,logger) 
        rec_losses.update(np.array([np.mean((all_out[v][all_ind[:,v]==0]-v_data[all_ind[:,v]==0])**2) for v,v_data in enumerate(all_ori)]).mean())
        exi_losses.update(np.array([np.mean((all_out[v][all_ind[:,v]==1]-v_data[all_ind[:,v]==1])**2) for v,v_data in enumerate(all_ori)]).mean())
        print('rec_loss:{:.4f}  exi_loss:{:.4f}'.format(rec_losses.vals[-1],exi_losses.vals[-1]))

    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))

    return losses,model,results,all_encX

def evaluate(H,all_label, estimator, classes_num, epoch,logger):
    
    end = time.time()
    preds = estimator.fit_predict(H)
    all_label = all_label.reshape(-1)
    results = clustering_metric(all_label,preds)
    print('ACC:{}  NMI:{}  PUR:{}'.format(results['ACC'],results['NMI'],results['PUR']))
    return results

    



def main(args,file_path):
    data_path = osp.join(args.data_dir,args.dataset+'.mat')
    fold_data_path = osp.join(args.fold_dir, args.dataset+'_percentDel_'+str(args.mask_view_ratio)+'.mat')\
        if args.dataset !='animal' else osp.join(args.fold_dir, args.dataset+'_pairedrate_'+str(args.mask_view_ratio)+'.mat')

    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(5)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' + 
                                    str(args.training_sample_ratio) + '_'+str(args.beta)+'_'+str(args.gamma)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    
    for fold_idx in range(folds_num):
        fold_idx=fold_idx+3
        train_dataloder,train_dataset = mydataset.getIncDataloader(data_path,fold_data_path,fold_idx=fold_idx,is_train=True,batch_size=args.batch_size,shuffle = False,num_workers=1)
        # train_dataloder,train_dataset = mydataset.getComDataloader(data_path,is_train=True,batch_size=args.batch_size,shuffle = False,num_workers=4)


        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        model = get_model(d_list,d_model=args.dim,n_layers=1,heads=4,classes_num=train_dataset.classes_num,dropout=0.)

        loss_model = myloss.MyLoss()
        # loss_model = nn.MSELoss(reduce=True, size_average=True)
        # crit = nn.BCELoss()
        # optimizer = nn.SGD(model.trainable_params(), learning_rate=args.lr, momentum=args.momentum)
        optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.85)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
        scheduler = None
        estimator = KMeans(n_clusters=classes_num, max_iter=300, n_init=10, random_state=928)
        
        logger.info('train_data_num:'+str(len(train_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        static_res = AverageMeter()
        epoch_results = [AverageMeter() for i in range(5)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        all_newX = []
        all_encX = np.ones((len(train_dataset),train_dataset.view_num,args.dim)).astype(np.float32)
        all_graph=None
        for epoch in range(args.epochs):
            if epoch<args.rec_epochs:
                train_losses,model,all_encX,all_newX = train_1(train_dataloder,train_dataset,model,all_graph,all_encX,loss_model,optimizer,scheduler,estimator,epoch,logger)
                all_graph = Tensor(getMvKNNGraph(all_newX,k=args.gamma),dtype = ms.float32)
                continue
            else:
                train_losses,model,evaluation_results,all_encX = train_2(train_dataloder,train_dataset,model,all_graph,all_encX,all_newX,loss_model,optimizer,scheduler,estimator,epoch,logger,fold_idx)
                
            
             
            
            
    
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--data-dir', type=str, metavar='PATH', 
                        default='data/')
    parser.add_argument('--fold-dir', type=str, metavar='PATH', 
                        default='data/')
    parser.add_argument('--dataset', type=str, default='handwritten-5view')#handwritten-5view NH_jerry Caltech101-7 animal
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--folds-num', default=1, type=int) 
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'output'))
    parser.add_argument('--img-dir', type=str, metavar='PATH', 
                        default='hw-imgs/0.5_')
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--save-img', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--name', type=str, default='final_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--rec-epochs', type=int, default=50)
    
    # Training args
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)
    # parser.add_argument('--r', type=float, default=1)
    
    
    args = parser.parse_args()
    file_path = osp.join(args.records_dir,args.name+str(args.epochs)+str(args.rec_epochs)+args.dataset+'_ViewMask_' + str(
                                    args.mask_view_ratio)+'.txt')
    args.file_path = file_path
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)

    args.lr = 1e-3
    args.beta = 1
    args.gamma = 15            
    main(args,file_path)