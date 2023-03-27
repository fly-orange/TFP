import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from typing import Union, Callable, Optional

#model
class Conv2D(nn.Module):
    r"""An implementation of the 2D-convolution block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." 
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        input_dims (int): Dimension of input.
        output_dims (int): Dimension of output.
        kernel_size (tuple or list): Size of the convolution kernel.
        stride (tuple or list, optional): Convolution strides, default (1,1).
        use_bias (bool, optional): Whether to use bias, default is True.
        activation (Callable, optional): Activation function, default is torch.nn.functional.relu.
        bn_decay (float, optional): Batch normalization momentum, default is None.
    """

    def __init__(self, input_dims: int, output_dims: int, kernel_size: Union[tuple, list], 
                stride: Union[tuple, list]=(1, 1), use_bias: bool=True, 
                activation: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]]=F.relu,
                bn_decay: Optional[float]=None):
        super(Conv2D, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                               padding=0, bias=use_bias)
        self._batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self._conv2d.weight)

        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the 2D-convolution block.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, input_dims).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, output_dims).
        """
        X = X.permute(0, 3, 2, 1)
        X = self._conv2d(X)
        X = self._batch_norm(X)
        if self._activation is not None:
            X = self._activation(X)
        return X.permute(0, 3, 2, 1)


class FullyConnected(nn.Module):
    """An implementation of the fully-connected layer.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." 
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        input_dims (int or list): Dimension(s) of input.
        units (int or list): Dimension(s) of outputs in each 2D convolution block.
        activations (Callable or list): Activation function(s).
        bn_decay (float, optional): Batch normalization momentum, default is None.
        use_bias (bool, optional): Whether to use bias, default is True.
    """

    def __init__(self, input_dims: Union[int, list], units: Union[int, list], 
                activations: Union[Callable[[torch.FloatTensor], torch.FloatTensor], list],
                bn_decay: float, use_bias: bool=True):
        super(FullyConnected, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        assert type(units) == list
        self._conv2ds = nn.ModuleList([Conv2D(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1],
            stride=[1, 1], use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the fully-connected layer.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, 1).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, units[-1]).
        """
        for conv in self._conv2ds:
            X = conv(X)
        return X

class SpatioTemporalEmbedding(nn.Module):
    r"""An implementation of the spatial-temporal embedding block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." 
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        D (int) : Dimension of output.
        bn_decay (float): Batch normalization momentum.
        steps_per_day (int): Steps to take for a day.
        use_bias (bool, optional): Whether to use bias in Fully Connected layers, default is True.
    """

    def __init__(self, D: int, bn_decay: float, steps_per_day: int, device, use_bias: bool=True):
        super(SpatioTemporalEmbedding, self).__init__()
        self._fully_connected_se = FullyConnected(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay, use_bias=use_bias)

        self._fully_connected_te = FullyConnected(
            input_dims=[steps_per_day+7, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay, use_bias=use_bias)
        self.device = device

    def forward(self, TE: torch.FloatTensor, T: int) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal embedding.

        Arg types:
            * **SE** (PyTorch Float Tensor) - Spatial embedding, with shape (num_nodes, D).
            * **TE** (Pytorch Float Tensor) - Temporal embedding, with shape (batch_size, num_his + num_pred, 2).(dayofweek, timeofday)
            * **T** (int) - Number of time steps in one day.

        Return types:
            * **output** (PyTorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_his + num_pred, num_nodes, D).
        """
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1).to(self.device)
        TE = TE.unsqueeze(dim=2)
        TE = self._fully_connected_te(TE)
        del dayofweek, timeofday
        return TE


## spatial-graph
class SpatialAttention(nn.Module):
    def __init__(self,
                 input_size: int,
                 q: int,
                 num_grulstm_layers: int,
                 device):
        
        super().__init__()
        self.num_grulstm_layers=num_grulstm_layers
        self.q=q
        self.device=device
        self.Wq = nn.GRU(input_size=input_size, hidden_size=q, num_layers=num_grulstm_layers,batch_first=True)
        self.Wk = nn.GRU(input_size=input_size, hidden_size=q, num_layers=num_grulstm_layers,batch_first=True)
#         self.Wv = nn.GRU(input_size=input_size, hidden_size=v, num_layers=num_grulstm_layers,batch_first=True)
#         self.Wo = nn.Linear(v,input_size)
        
    # input X: (B, T , N, D)
    # output Y: (B, T, N, D)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        K = self.q
        x1= torch.squeeze(torch.cat(x.chunk(x.shape[2],dim=2),dim=0)) #(B*N, T， D)

        _ , Wq1 = self.Wq(x1)  #（1，B*N，D）  
        _ , Wk1 = self.Wk(x1)
        Wq=torch.cat(Wq1.transpose(0,1).chunk(x.shape[2],dim=0),dim=1) #（B, N，D）
        Wk=torch.cat(Wk1.transpose(0,1).chunk(x.shape[2],dim=0),dim=1) #（B, N, D）
#             _,Wv[:,i:i+1,:] = self.Wv(x[:,:,i:i+1],hidden_init_v)

        score = torch.bmm(Wq,Wk.transpose(1,2))/np.sqrt(K)
        
#         future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
#         future_mask = future_mask.to(score.device)
#         score = score.masked_fill(future_mask, float('-inf'))
        
        attention = F.softmax(score,dim=-1)  # (B, N, N)
        y = torch.matmul(attention.unsqueeze(1),x)
        return y

class TemporalAttention(nn.Module):
    r"""An implementation of the temporal attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction." 
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
        mask (bool): Whether to mask attention score.
    """

    def __init__(self, K: int, d: int, bn_decay: float, mask: bool,device):
        super(TemporalAttention, self).__init__()
        D = K * d
        self.device = device
        self._d = d
        self._K = K
        self._mask = mask
        self._fully_connected_q = FullyConnected(input_dims= D, units=D, activations=F.relu,
                                                 bn_decay=bn_decay)
        self._fully_connected_k = FullyConnected(input_dims= D, units=D, activations=F.relu,
                                                 bn_decay=bn_decay)
        self._fully_connected_v = FullyConnected(input_dims= D, units=D, activations=F.relu,
                                                 bn_decay=bn_decay)
        self._fully_connected = FullyConnected(input_dims=D, units=D, activations=F.relu,
                                               bn_decay=bn_decay)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the temporal attention mechanism.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **STE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Temporal attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        query = self._fully_connected_q(X)
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._d, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= (self._d ** 0.5)
        if self._mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_nodes = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = mask.unsqueeze(0).unsqueeze(0).to(torch.bool).to(self.device)
            mask = mask.repeat(self._K * batch_size, num_nodes, 1, 1)
            condition = torch.FloatTensor([-2 ** 15 + 1]).to(self.device)
            attention = torch.where(mask, attention, condition)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._fully_connected(X)
        del query, key, value, attention
        return X

    
class EncoderRNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_grulstm_layers):
        super(EncoderRNN, self).__init__()
        self.num_grulstm_layers=num_grulstm_layers
        self.hidden_size=hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]      
        output, hidden = self.gru(input, hidden)      
        return output, hidden    

    # def init_hidden(self,device):
    #     #[num_layers*num_directions,batch,hidden_size]   
    #     return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device) 


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers):
        super(DecoderRNN, self).__init__()      
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)        

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)       
        return output, hidden
    
        
class GraphTrans(nn.Module):
    def __init__(self,args):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        D = args.gt_D
        K = args.gt_K  
        bn_decay = args.bn_decay
        
        self.D = D
        self.K = K
        self.p_length = args.p_length
        self.device = args.device
        ##block
        self._fully_connected_1 = FullyConnected(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                                                 bn_decay=bn_decay)
        self._fully_connected_2 = FullyConnected(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                                                 bn_decay=bn_decay)
        self._st_embedding = SpatioTemporalEmbedding(D =D, bn_decay = bn_decay, steps_per_day = 288 ,device = self.device, use_bias= False)


        self.spatial_encoder = SpatialAttention(D,D,1,self.device)
        
        self.encoder = EncoderRNN(2*D,D,1)
        self.decoder = DecoderRNN(D,D,1) 
        
        self.temporal_decoder = TemporalAttention(K, (D//K), bn_decay, True, self.device)


    def forward(self, trainX: torch.Tensor, trainZ : torch.Tensor)->torch.Tensor:
        '''
        Making a forward pass of GT.
        Raw Input:
            trainX.shape : (B, T, N, 3) 
            trainZ.shape : (B, T, N, 2)
        
        Transformed Input:
            X.shape : (B, T, N)
            TE.shape :  (B, T+T, 2)
        '''
        inputX = trainX[...,0:1]
        X = self._fully_connected_1(inputX)  # (B, T_his, N, D)

        N = X.shape[2]
        TE = torch.cat((trainX[:,:,0,1:],trainZ[:,:,0,:]),axis=1) #(B, 2T, 2)

        #spatial encoder
        T_his = X.shape[1]
        T_pred = TE.shape[1] - X.shape[1]
        X_embedding = torch.cat([ self.spatial_encoder(X[:,i*self.p_length:i*self.p_length+self.p_length,:]) for i in range(X.shape[1]//self.p_length)],dim = 1)  # (B, T_his, N, D)
        
        TE_embedding = self._st_embedding(TE,288).repeat(1,1,N,1)  # (B, his + pred, N, D)
        TE_his = TE_embedding[:, :X.shape[1]]  # (B, T_his, N, D)
        TE_pred = TE_embedding[:, X.shape[1]:]    # (B, T_pred, N, D)

        X_his = torch.cat((X_embedding , TE_his),dim = -1)  #(B, T_his, N, 2*D)
        
        ##temporal encoder
        X_his = torch.squeeze(torch.cat(X_his.chunk(N,dim=2),dim=0)) #(B*N, T， D)

        encoder_hidden = torch.zeros(1, X_his.shape[0], self.D, device=self.device)
        his_embedding = torch.zeros([X_his.shape[0], T_his, self.D]).to(self.device) # (B*N, T_his, D)
        for ei in range(T_his):
            encoder_output, encoder_hidden = self.encoder(X_his[:,ei:ei+1,:] , encoder_hidden)
            his_embedding[:,ei:ei+1,:] = encoder_output
        his_embedding = his_embedding.unsqueeze(2) # (B*N, T_his, 1,D)
        his_embedding = torch.cat(his_embedding.chunk(N,dim=0),dim=2) # (B, T_his, N, D)

        decoder_hidden = encoder_hidden
        pred_embedding = torch.zeros([X_his.shape[0], T_pred, self.D]).to(self.device) # (B*N, T_pred, D)
        X_pred = torch.squeeze(torch.cat(TE_pred.chunk(N,dim=2),dim=0)) #(B*N, T， D)
        for di in range(T_pred):
            decoder_output, decoder_hidden = self.decoder(X_pred[:,di:di+1,:]  , decoder_hidden)
            pred_embedding[:,di:di+1,:] = decoder_output  
        pred_embedding = pred_embedding.unsqueeze(2) # (B*N, T, 1, D)
        pred_embedding = torch.cat(pred_embedding.chunk(N,dim=0),dim=2) # (B, T_pred, N, D)
        embedding = torch.cat([his_embedding,pred_embedding],axis=1)  # (B, T_his + T_pred, N, D)
        
        ##temporal decoder
        embedding2 = self.temporal_decoder(embedding) # ((B, T_his + T_pred, N, D))
        future_decoding = embedding2[:,T_his:]  # ((B, T_pred, N, D))
        
        ##prediction net
        y = torch.squeeze(self._fully_connected_2(future_decoding),3)  # (B, T_pred, N)

        return y
    