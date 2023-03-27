'''Control Training, Evaluation and Testing'''
import logging
import numpy as np
import torch
import copy
from utils.name_match import *
from utils.utils import AverageMeter, metrics


class trainer():
    def __init__(self, params, scaler):
        self.model = model_objects[params.model](params).to(params.device)
        if params.pretrain:
            pretrain_file = params.save_path + '/' + params.pretrain_file
            self.model.load_state_dict(torch.load(pretrain_file))
            logging.info(f'model has been loaded from {pretrain_file} ')
        self.criterion = criterion_objects[params.criterion]
        self.optimizer = opt_objects[params.optimizer](self.model.parameters(), 
                                                       lr = params.lr, weight_decay = params.weight_decay)
        self.scheduler = sche_object[params.scheduler](self.optimizer, 
                                                       step_size = params.decay_epoch, gamma = 0.9 )
        self.scaler = scaler
        self.params = params
        

    def train(self, dataloader):
        self.model.train()

        loss_train = AverageMeter()
        rmse_train = AverageMeter()
        mae_train = AverageMeter()
        mape_train = AverageMeter()

        for batch_idx, data in enumerate(dataloader):
            # x.shape (B, T, N, 2) historical information
            # y.shape (B, T, N, 1) future label
            # z.shape (B, T, N, 1 or 2)  future time information

            his, future = data
            x, y, z = his, future[..., 0], future[..., 1:]
            x, y, z = x.cuda(), y.cuda(), z.cuda()
            # out.shape (B, T, N, 1)
            # X = copy.deepcopy(x[...,0])  #(B, T, N,)
            # TE = torch.cat((x[:,:,0,1:],z[:,:,0,:]),axis=1) #(B, 2T, 2)
            # X, TE, y = X.cuda(), TE.cuda(), y.cuda()
            self.optimizer.zero_grad()
            out = self.model(x,z)
            pred = self.scaler.inverse_transform(out)
            loss_batch = self.criterion(pred, y) 
            loss_batch.backward()
            loss_train.update(loss_batch, self.params.batch_size)
            self.optimizer.step()
            if (batch_idx+1) % self.params.print_every_iter ==0:
                print(f'Iter: {batch_idx+1:03d}, train loss:{loss_batch:.4f}')
            
            rmse, mae, mape = metrics(pred, y)
            rmse_train.update(rmse, self.params.batch_size)
            mae_train.update(mae, self.params.batch_size)
            mape_train.update(mape, self.params.batch_size)
            
            # del data

        if self.params.lr_decay: # wheather to update lr
            self.scheduler.step()

        return loss_train.avg(), rmse_train.avg(), mae_train.avg(), mape_train.avg()
    
    def eval(self, dataloader):
        self.model.eval()
        
        loss_eval = AverageMeter()
        rmse_eval = AverageMeter()
        mae_eval = AverageMeter()
        mape_eval = AverageMeter()
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                his, future = data
                x, y, z = his, future[..., 0], future[..., 1:]
                x, y, z = x.cuda(), y.cuda(), z.cuda()
                
                out = self.model(x,z)
                pred = self.scaler.inverse_transform(out)
                loss_batch = self.criterion(pred, y)
                loss_eval.update(loss_batch, his.shape[0])
                # evaluation metrics
                rmse_eval_batch, mae_eval_batch, mape_eval_batch = metrics(pred, y)
                rmse_eval.update(rmse_eval_batch, his.shape[0])
                mae_eval.update(mae_eval_batch, his.shape[0])
                mape_eval.update(mape_eval_batch, his.shape[0])

        return loss_eval.avg(), rmse_eval.avg(), mae_eval.avg(), mape_eval.avg()


class tester():
    def __init__(self, params, scaler):
        self.params = params
        self.scaler = scaler

    def predict(self, model, dataloader):
        model.eval()
        pred = []
        real = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                his, future = data
                x, y, z = his, future[...,0], future[...,1:]
                x, y, z = x.cuda(), y.cuda(), z.cuda()
                pred_batch = model(x, z)
                pred.append(pred_batch.cpu().detach().clone())
                real.append(y.cpu().detach().clone())
        pred = torch.from_numpy(np.concatenate(pred, axis =0)) 
        pred = self.scaler.inverse_transform(pred)  # (B, T, N, 1)
        real = torch.from_numpy(np.concatenate(real, axis =0)) 
        return pred, real

    def eval(self, pred, real):
        total_rmse, total_mae, total_mape = metrics(pred, real)
        logging.info('                MAE\t\tRMSE\t\tMAPE')
        logging.info('test             %.2f\t\t%.2f\t\t%.2f%%' %
                (total_mae, total_rmse, total_mape * 100))
        logging.info('performance in each prediction step')
        MAE, RMSE, MAPE = [], [], []
        for step in range(self.params.output_length):
            rmse, mae, mape = metrics(pred[:, step], real[:, step])
            RMSE.append(rmse)
            MAE.append(mae)
            MAPE.append(mape)
            logging.info('step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                    (step + 1, mae, rmse, mape * 100))
        average_mae = np.mean(MAE)
        average_rmse = np.mean(RMSE)
        average_mape = np.mean(MAPE)
        logging.info('average:         %.2f\t\t%.2f\t\t%.2f%%' %
                (average_mae, average_rmse, average_mape * 100))
        return RMSE, MAE, MAPE
