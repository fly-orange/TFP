'''
Training a GNN model for TFP Task 
Code adpated from 
https://github.com/nnzhan/Graph-WaveNet.git &
https://github.com/VincLee8188/GMAN-PyTorch.git

1. Load data
2. Load model
3. Train a model
4. Evaluate and save 
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 必须放在所用用到gpu指令之前
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import numpy as np
import argparse
import time
import torch
import torch.optim as optim
from utils.utils import *
from utils.name_match import *
from engine import *
from tensorboardX import SummaryWriter 

''' Argument 
==================================='''
parser = argparse.ArgumentParser()

# Main Hyperparamter
parser.add_argument('--device', type = str, default = 'cuda:0',
                    help='')
parser.add_argument('--onlytest', action = 'store_true',
                    help = 'wheather only test a model')
parser.add_argument('--pretrain', action = 'store_true', 
                    help = 'wheather to use a pretrained model')
parser.add_argument('--pretrain_file', type = str, default = 'model_exp1_best.pth',
                    help = '')
parser.add_argument('--expid', type = int, default = 1,
                    help = '')

# Data Hyperparameter
parser.add_argument('--data', type = str, default = 'PEMS-BAY',
                    choices = ['PEMS-BAY', 'METR-LA'])
parser.add_argument('--input_length', type = int, default = 12,
                    help = '')
parser.add_argument('--output_length', type = int, default = 12,
                    help = '')
parser.add_argument('--dow', action = 'store_true',
                    help = 'Whether to take day of week as input')
                                       
# Model Hyperparamter
parser.add_argument('--model', type = str, default = 'GT',
                    choices = ['GMAN', 'GWN', 'GT'])
parser.add_argument('--criterion', type = str, default = 'MAE',
                    choices =['MSE', 'L1','MAE'])
parser.add_argument('--optimizer', type = str, default = 'Adam',
                    choices = ['Adam', 'SGD'])
parser.add_argument('--scheduler', type = str, default = 'step',
                    choices = ['step', 'exp', 'cos'])

# Training Hyperparamter
parser.add_argument('--epochs', type = int, default = 100,
                    help='training epochs')
parser.add_argument('--batch_size', type = int, default = 64,
                    help='')
parser.add_argument('--lr', type = float, default = 0.001,
                    help = 'learning rate')
parser.add_argument('--weight_decay', type = float, default = 0.0001,
                    help = 'weight decay')              
parser.add_argument('--lr_decay',type = bool,default = True,
                    help='wheather to decrease learning rate')                    
parser.add_argument('--decay_epoch',type = int,default = 10,
                    help='epochs of learning rate decay')
parser.add_argument('--print_every_iter', type = int, default = 50,
                    help='Iters to print training loss')                  
parser.add_argument('--patience', type = int, default = 5,
                    help='epochs to control early stop')

# Specific Model Hyperparamters
'''Graph WaveNet'''
parser.add_argument('--dropout',type=float,default=0.3,help='')
parser.add_argument('--gcn_bool',type=bool,default=True,help='whether to add GCN'  )
parser.add_argument('--adjprior',type=bool,default=True,help='whether use prior adj')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='prior adj type')
parser.add_argument('--adjadapt',type=bool,default=True,help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--nhid',type=int,default=32,help='')

'''GMAN'''
parser.add_argument('--gm_L',type=int,default=3,help='number of STAtt Blocks')
parser.add_argument('--gm_K',type=int,default=8,help='number of attention heads')
parser.add_argument('--gm_d',type=int,default=8,help='dims of each attention head')
parser.add_argument('--gm_use_bias', type=bool, default=True)
parser.add_argument('--mask', action='store_true', help='whether add temporal mask')
parser.add_argument('--bn_decay', type=float, default=0.1)
parser.add_argument('--steps_per_day', type=int, default=288)

'''GT'''
parser.add_argument('--p_length',type=int,default=6,help='length of p window')
parser.add_argument('--gt_K',type=int,default=8,help='number of attention heads')s
parser.add_argument('--gt_D',type=int,default=64,help='hidden dim')


# Argument Supplementary
args = parser.parse_args()

args.device = torch.device(args.device)
args.num_nodes = nodes_objects[args.data]
args.save_path = f'result/{args.data}/{args.model}'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path) # makedirs多层创建目录

# Specific Model Hyperparameter
'''Graph WaveNet'''
if args.adjprior: ## whether use prior adj
    adj_file = f'data/{args.data}/adj_mx.pkl'
    sensor_ids, sensor_id_to_ind, adj_mx = load_adj(adj_file,args.adjtype) 
    args.supports = [torch.tensor(i).to(args.device) for i in adj_mx] # aggregation matrix set from adj
else:
    args.supports = None

if args.randomadj:
    args.adjinit = None
else:
    args.adjinit = args.supports[0]

'''GMAN'''
SE_file = f'data/{args.data}/SE.txt'
args.SE = SE(SE_file).to(args.device)

''' Components (data + train + test)
======================================================'''

def load_data():
    '''Load data'''
    logging.info('Loading Data')

    data_file = data_files[args.data]
    data = generate_dataloader(data_file, args)
    logging.info('Data Loaded')
    return data

def train(data):
    '''Load Model'''
    logging.info('Loading Model')
    engine = trainer(args, data['scaler'])
    writer = SummaryWriter('logs/metr')

    param_num = count_parameters(engine.model)
    logging.info('Total number of model parameters is {}'.format(param_num))
    logging.info('Model Loaded')
   
    '''Training'''
    logging.info('***** training model *****')

    loss_train_list = []
    loss_val_list = []
    loss_test_list = []
    wait = 0

    if args.pretrain:    # if pretrain: loss_val_min 继承 pretrained model
        loss_val_init,_ ,_ ,_= engine.eval(data['val_loader'])
        loss_val_min = loss_val_init
        logging.info(f'Initial val loss : {loss_val_min}')
    else:
        loss_val_min = float('inf')
    
    # data['train_loader'].shuffle()
    for ep in range(args.epochs):
        # Training 
        t0 = time.time()
        logging.info(f'Epoch {ep+1:03d} ')
        # data['train_loader'].shuffle()
        loss_train_epoch, rmse_train_epoch, mae_train_epoch, mape_train_epoch = engine.train(data['train_loader'])
        loss_train_list.append(loss_train_epoch)
        
        # Evaluation
        t1 = time.time()
        loss_val_epoch, rmse_val_epoch, mae_val_epoch, mape_val_epoch = engine.eval(data['val_loader'])
        loss_val_list.append(loss_val_epoch)
        
        # Print to log and terminal
        t2 = time.time()
        logging.info(f'Epoch {ep+1:03d}: train time : {t1-t0:.4f}, inference time : {t2-t1:.4f}')
        logging.info(f'Epoch {ep+1:03d}: train loss: {loss_train_epoch:.4f}, train rmse: {rmse_train_epoch:.4f}, train mae: {mae_train_epoch:.4f}, train mape: {mape_train_epoch:.4f}')
        logging.info(f'Epoch {ep+1:03d}: val loss: {loss_val_epoch:.4f}, val rmse: {rmse_val_epoch:.4f}, val mae: {mae_val_epoch:.4f}, val mape: {mape_val_epoch:.4f}')
        
        # Tensorboard
        writer.add_scalar('Train-Loader/Loss', loss_train_epoch, global_step = ep+1)
        writer.add_scalar('Val-Loader/Loss', loss_val_epoch, global_step = ep+1)

        # Early stop
        best_epoch = 0
        if loss_val_epoch <= loss_val_min :
            logging.info(f'val loss decrease from {loss_val_min:.4f} to {loss_val_epoch:.4f}')
            loss_val_min = loss_val_epoch
            wait = 0 
            best_epoch = ep+1
            best_model_wts = engine.model.state_dict()
            torch.save(best_model_wts, args.save_path + f'/model_exp{args.expid}_best.pth')
        else:
            wait += 1
            if wait >= args.patience:
                logging.info(f'Early stop at epoch : {ep+1:03d}')
                break

    logging.info(f'Model achives its peak at epoch {best_epoch} with val loss {loss_val_min}')
    logging.info('***** Finishing Training *****')
    engine.model.load_state_dict(best_model_wts)

    return engine.model

def test(model, data):
    '''test model'''  
    logging.info('***** Testing Performance *****')
    test_engine = tester(args, data['scaler'])
    test_pred, test_real = test_engine.predict(model, data['test_loader'])
    test_rmse, test_mae, test_mape = test_engine.eval(test_pred, test_real)

''' Main funtion
=============================================='''

if __name__ == '__main__': 
    init_logging('logs/logfile.log')
    start_time = time.time()
    data = load_data()
    
    if not args.onlytest:
        final_model = train(data)
    else:
        final_model = model_objects[args.model](args).to(args.device)
        model_file = args.save_path + f'/model_exp{args.expid}_best.pth'
        final_model.load_state_dict(torch.load(model_file))

    test(final_model, data)
    end_time = time.time()
    print("Total time spent: {:.4f}".format(start_time-end_time))

