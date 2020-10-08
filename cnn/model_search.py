import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module): #求得混合操作的结果

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()  #nn.ModuleList()是储存操作的容器，没有先后顺序
    for primitive in PRIMITIVES:  #Primitive定义的是8个操作
      op = OPS[primitive](C, stride, False) #在operation.py中有OPS的定义
      #其中False指定batchnorm的使用不更新
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))  #nn.Sequential默认按照顺序执行 即将pool操作和batchnorm打包操作
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops)) #没有先后顺序 输入x，求得Σo(x)，所有操作的和


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__() #C_pre_prev表示第k-2个cell的输出通道数
    self.reduction = reduction #C_prev表示第k-1个cell的输出通道数，C表示当前cell的输出通道数
    if reduction_prev:#如果之前的一个cell是reduction cell 则需要减半
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)#定义节点0的操作
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)#定义节点1的操作
    #开始的两个节点将通道数调整为C，后面的操作均保持通道数不变
    self._steps = steps #一共有4个节点需要优化
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):#首先定义好操作
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1  ##size只缩小为两倍，只需要前两个操作的stride为2即可
        op = MixedOp(C, stride)#先初始化每个节点的8个操作 每个节点都是8个操作混合起来
        self._ops.append(op)#得到初始化的操作 len=14
        '''
        self._ops[0,1]表示内部节点2的前继操作
        self._ops[2,3,4]表示内部节点3的前继操作
        self._ops[5,6,7,8]表示内部节点4的前继操作
        self._ops[9,10,11,12,13]表示内部节点5的前继操作
        '''

  def forward(self, s0, s1, weights): #给定权重进行前向传递 weights为14*8大小的矩阵
    s0 = self.preprocess0(s0)#s0来自k-2个cell的输出
    s1 = self.preprocess1(s1)#s1来自k-1个cell的输出

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))#对每一个节点，求其前继结点经过操作后与对应操作权重的加权值
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)#将他们在channel层拼接在一起，形成cell的输出 取state中最后四项


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    #c =16 num_class = 10,layers = 8 
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier#4,因为有4个中间节点，所以通道最后扩大四倍

    C_curr = stem_multiplier*C #48
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )#起始的操作 将图像卷积，通道数扩大为48
    
    #对于第一个cell，其k-2和k-1个cell都是stem
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]: #返回第2个和第5个为reduction cell
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction  
       '''
    layers = 8, 第2和5个cell是reduction_cell
    cells[0]: cell = Cell(4, 4, 48,  48,  16, false,  false) 输出[N,16*4,h,w]
    cells[1]: cell = Cell(4, 4, 48,  64,  16, false,  false) 输出[N,16*4,h,w]
    cells[2]: cell = Cell(4, 4, 64,  64,  32, True,   false) 输出[N,32*4,h/2,w/2]
    cells[3]: cell = Cell(4, 4, 64,  128, 32, false,  false) 输出[N,32*4,h/2,w/2]  #当采样cells[1]的结果时，要将feature map大小变为二分之一才能匹配。
    cells[4]: cell = Cell(4, 4, 128, 128, 32, false,  false) 输出[N,32*4,h/2,w/2]
    cells[5]: cell = Cell(4, 4, 128, 128, 64, True,   false) 输出[N,64*4,h/4,w/4]
    cells[6]: cell = Cell(4, 4, 128, 256, 64, false,  false) 输出[N,64*4,h/4,w/4]
    cells[7]: cell = Cell(4, 4, 256, 256, 64, false,  false) 输出[N,64*4,h/4,w/4]
    '''
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr
    
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    #定义分类器
    self._initialize_alphas()

  def new(self): #拷贝相同的alpha参数
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data) #拷贝相同的网络结构参数
    return model_new

  def forward(self, input): #正向传播 经过8个cell最后输出平均池化
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)#横着使用softmax求概率
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)#全局平均池化
    logits = self.classifier(out.view(out.size(0),-1))#分类器
    return logits#输出结果

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):#初始化网络结构参数
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    # k=14
    num_ops = len(PRIMITIVES) #num_ops=8

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    #(14,8)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      #weights[14,8]
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy() #获得当前中间节点关于所有前继节点的权重
        #实际为2*8的权重矩阵
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]#返回权重最大的两个连接节点
        #返回具有最大权重边的那个前继节点 按照倒叙排列
        #定义死了只返回两个最大的edge对应的节点 即返回两个具有最大权重边的节点
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))#挑出每个节点对应的最大权重的操作
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)#concat的节点是事先指定好 [2,3,4,5]
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

