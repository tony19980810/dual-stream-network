import os

import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import scipy.io as scio
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def gen_feature():
    """
    gain node matrix
    """
    for root, dirs, files in os.walk(os.path.join('.', 'data', 'MEIP_au_O_A_30')):
        for j in files:
            feature=np.zeros(510)
            csv_path=os.path.join('.', 'data', 'MEIP_au_O_A_30',j)
            df = pd.read_csv(csv_path, header=None)
            df = df.values
            l=0
            for k in range(0, 30):
                for i in range(2,19):
                    feature[l]=df[k,i]
                    l=l+1
            scio.savemat(
                os.path.join('.', 'data', 'MEIP_au_O_A_30_flatten', j[:-4]+'.mat'),
                {'data': feature})

def gen_au_adj():
    """
        initialize the 3D adjacency matrix
    """
    if not os.path.exists(os.path.join('.', 'adj', 'train')):
        os.makedirs(os.path.join('.','adj','train'))
    if not os.path.exists(os.path.join('.','adj','test')):
        os.makedirs(os.path.join('.','adj','test'))
    for i in (2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24):
        f = open(os.path.join('.','adj_txt','train',str(i)+'.txt'))
        adj = np.zeros((510, 510))
        readline = f.readline()
        count=np.zeros(17)
        while readline:
            tmp = readline.split(' ')
            csv_path=tmp[0]
            df=pd.read_csv(csv_path,header=None)
            df=df.values
            for j in range(0, 30):
                tmp = []
                for q in range(19,37):
                    if q==35:
                        continue
                    if int(df[j,q])==1:
                        if q<35:
                            count[q-19]=count[q-19]+1
                            tmp.append(q-19)
                        else:
                            count[16]=count[16]+1
                            tmp.append((16))
                if len(tmp)>=2:
                    for k in tmp:
                        for l in tmp:
                            if l==k:
                                continue
                            else:
                                adj[k,l]=adj[k,l]+1

            readline = f.readline()
        f.close
        for n in range(0, 17):
            for m in range(0, 17):
                if adj[n, m] == 0:
                    continue
                else:
                    adj[n, m] = float(adj[n, m]) / float(count[m])
        tmp=[]
        for z in range(1,30):
            tmp.append(z*17)
        for j in tmp:
            for k in range(j,j+17):
                for l in range(j,j+17):
                    adj[k,l]=adj[k%17,l%17]
        for k in range(0,17):
            adj[k,k+17]=1
        for k in range(17,493):
            adj[k,k+17]=1
            adj[k,k-17]=1
        for k in range(493,510):
            adj[k,k-17]=1
        scio.savemat(
            os.path.join('.', 'adj','train', str(i) + '.mat'),
            {'data': adj})
        print(count)
        f = open(os.path.join('.', 'adj_txt', 'test', str(i) + '.txt'))
        adj = np.zeros((510, 510))
        readline = f.readline()
        count = np.zeros(17)
        while readline:
            tmp = readline.split(' ')
            csv_path = tmp[0]
            df = pd.read_csv(csv_path, header=None)
            df = df.values
            for j in range(0, 30):
                tmp = []
                for q in range(19, 37):
                    if q == 35:
                        continue
                    if int(df[j, q]) == 1:
                        if q < 35:
                            count[q - 19] = count[q - 19] + 1
                            tmp.append(q - 19)
                        else:
                            count[16] = count[16] + 1
                            tmp.append((16))
                if len(tmp) >= 2:
                    for k in tmp:
                        for l in tmp:
                            if l == k:
                                continue
                            else:
                                adj[k, l] = adj[k, l] + 1

            readline = f.readline()
        f.close
        for n in range(0, 17):
            for m in range(0, 17):
                if adj[n, m] == 0:
                    continue
                else:
                    adj[n, m] = float(adj[n, m]) / float(count[m])
        tmp = []
        for z in range(1, 30):
            tmp.append(z * 17)
        for j in tmp:
            for k in range(j, j + 17):
                for l in range(j, j + 17):
                    adj[k, l] = adj[k % 17, l % 17]
        for k in range(0, 17):
            adj[k, k + 17] = 1
        for k in range(17, 493):
            adj[k, k + 17] = 1
            adj[k, k - 17] = 1
        for k in range(493, 510):
            adj[k, k - 17] = 1
        scio.savemat(
            os.path.join('.', 'adj', 'test', str(i) + '.mat'),
            {'data': adj})

def gen_au_adj_mask():
    """
    gain mask matrix
    """
    adj = np.zeros((510, 510))
    tmp = []
    for z in range(0, 30):
        tmp.append(z * 17)
    for j in tmp:
        for k in range(j, j + 17):
            for l in range(j, j + 17):
                adj[k, l] = 1
    scio.savemat(
        os.path.join('.',  'mask.mat'),
        {'data': adj})










