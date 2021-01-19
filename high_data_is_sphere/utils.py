import numpy as np
import torch


# 输入为state_dict
# 将weight和bias分开统计，输出为l2范数，l1范数


def para_stat(sd):
    w = list()
    b = list()
    keys = list(sd.keys())
    for key in keys:
        key_attr = key.split('.')[-1]
        if key_attr == 'weight':
            w.append(sd[key].reshape(-1))
        elif key_attr == 'bias':
            b.append(sd[key].reshape(-1))
    w = np.round(torch.cat(w).numpy(), 3)
    b = np.round(torch.cat(b).numpy(), 3)
    w1 = np.linalg.norm(w, ord=1)
    w2 = np.linalg.norm(w, ord=2)
    b1 = np.linalg.norm(b, ord=1)
    b2 = np.linalg.norm(b, ord=2)
    return w1, w2, b1, b2


# 输入为模型本身
# 输出为梯度的L1, L2范数以及sum求和

def grad_stat(model):
    temp = list()
    for p in model.parameters():
        temp.append(p.grad.reshape(-1))
    temp = np.round(torch.cat(temp).numpy(), 3)
    l1 = np.linalg.norm(temp, ord=1)
    l2 = np.linalg.norm(temp, ord=2)
    s = np.sum(temp)
    return l1, l2, s











