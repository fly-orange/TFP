# Graph Transformer for Traffic Flow Prediction

There exists three ways to constitute aggregation matrix of GCN:
(1) Normalized adj
(2) Transition matrix from adj
(3) self-adaptive matrix 

## 各模型复现指令
1. GWN： python main.py
2. GMAN: python main.py --dow --batch_size 32


## GWN复现难点
1. loss function使用null参数为0的masked mae
2. 自定义dataloader dataloader的shuffle在each epoch都会进行
3. 将self.变量赋给另一变量是危险的，容易造成self不必要的更改
4. 每次保存模型不要保存整个模型，一是加载不方便，二是容易更改模型的训练轨迹
5. model.eval()会影响评价结果， with torch.no_grad()会影响计算空间和时间

## GMAN复现难点
1. 必须使用小batch_size, GMAN对显存要求大
2. 内存泄露，使用memory profile检查内存增大的源头
 from memory_profiler import profile
 fp = open('.log','w+')
 @profile(stream = fp)
 泄露源头：mask.repeat(),将mask先放到gpu上可以解决此问题？
 3. dataloader最好要shuffle，数据某个batch loss为0， 在valid loss上误差高
 4. 