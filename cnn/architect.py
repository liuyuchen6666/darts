import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):#将向量全部展开为一维
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):
#计算alpha的梯度
  def __init__(self, model, args):
    self.network_momentum = args.momentum #设置动量
    self.network_weight_decay = args.weight_decay#设置权重衰减正则化
    self.model = model 
    #设置优化器训练alpha
    #设置优化器仅优化arch_parameter
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    # w − ξ*dwLtrain(w, α)
    #需要先更新一步w，然后根据当前的w取值来优化alpha
    loss = self.model._loss(input, target)#计算训练集的前向误差
    theta = _concat(self.model.parameters()).data #将权重展开为向量形式
    try:#上一步缓存的动量乘以momentum系数
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    #求得动量
    #moment = mu * v_{t} 
    #dw = g_{t+1} + weight_decay * w 
    #v_{t+1} = moment + dw
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    #self._construct_model_from_theta ?
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))# w − ξ*dwLtrain(w, α)
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:#如果不展开即相当于first order近似，直接求val loss进行更新
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    #首先更新一次权重系数w，得到w'
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)#L_val(w')
    #更新之后计算误差
    unrolled_loss.backward()#计算更新alpha的值
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]# dα Lval(w',α)
    #计算关于alpha的梯度
    vector = [v.grad.data for v in unrolled_model.parameters()]# dw'Lval(w',α)
    #计算关于权重系数的梯度
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)# 计算(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon) 
    #计算最后的梯度 gradient = dalpha - eta*implicit_grads
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)  #更新alpha的梯度为g

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data) #将计算得到的梯度值传给模型参数更新
      else:
        v.grad.data.copy_(g.data)#返回给定的对于alpha求得的梯度值

  def _construct_model_from_theta(self, theta):
    #根据权重系数建立新模型
    model_new = self.model.new() #创立新模型
    model_dict = self.model.state_dict()
    
    # 按照之前的大小，copy  theta参数
    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()#.norm()?eps
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v) #w = w+eps*dw
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    #求得海森矩阵
    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

