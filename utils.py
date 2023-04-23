import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def generate_txt_path(round:int):
    """
    gain dataset txt path
    """
    train_txt_path=os.path.join('.','txt','train',str(round)+'.txt')
    test_txt_path=os.path.join('.','txt','test',str(round)+'.txt')
    return train_txt_path,test_txt_path
def setup_seed(seed):
    """
    set seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)
    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)
    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer1,optimizer2, data_loader, device, epoch,round):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, gcns,labels= data
        sample_num += images.shape[0]
        pred = model(images.to(device),gcns.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "round:{},[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(round,epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch,round):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    matrix=[]
    for step, data in enumerate(data_loader):
        images, gcns,labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device),gcns.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        for i in range(0,len(pred_classes)):
            matrix.append([int(pred_classes[i]),int(labels[i])])
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "round:{},[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(round,epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,matrix,pred_classes,labels

def calculate_acc(matrix):
    """
    caculate accuracy
    """
    sample_num=len(matrix)
    right_num=0
    for i in matrix:
        if i[0]==i[1]:
            right_num=right_num+1
    return right_num,sample_num,float(right_num)/float(sample_num)
def confusion_matrix(matrix, conf_matix):
    """
    gain confusion matrix
    """
    for i in matrix:
        pl,tl = i
        tl = int(tl)
        pl = int(pl)
        conf_matix[tl, pl] = conf_matix[tl, pl] + 1
    return conf_matix
def calculate_prediction(metrix):
    """
    caculate prediction
    """
    label_pre = []
    current_sum = 0
    for i in range(metrix.shape[0]):
        current_sum += metrix[i][i]
        label_total_sum = metrix.sum(axis=0)[i]
        pre = round(100 * metrix[i][i] / label_total_sum, 4)
        label_pre.append(pre)
    print("每类精度：", label_pre)
    all_pre = round(100 * current_sum / metrix.sum(), 4)
    print("总精度：", all_pre)
    return label_pre, all_pre
def calculate_recall(metrix):
    """
    caculate recall
    """
    label_recall = []
    for i in range(metrix.shape[0]):
        label_total_sum = metrix.sum(axis=1)[i]
        label_correct_sum = metrix[i][i]
        recall = 0
        if label_total_sum != 0:
            recall = round(100 * float(label_correct_sum) / float(label_total_sum), 4)
        label_recall.append(recall)
    print("每类召回率：", label_recall)
    all_recall = round(np.array(label_recall).sum() / metrix.shape[0], 4)
    print("总召回率：", all_recall)
    return label_recall, all_recall
def calculate_f1(prediction, all_pre, recall, all_recall):
    """
    caculate f1
    """
    signal_f1 = []
    for i in range(len(prediction)):
        pre, reca = prediction[i], recall[i]
        f1 = 0
        if (pre + reca) != 0:
            f1 = round(2 * pre * reca / (pre + reca), 4)

        signal_f1.append(f1)
    print("每类f1：", signal_f1)
    all_f1 = round(2 * all_pre * all_recall / (all_pre + all_recall), 4)
    print("总的f1：", all_f1)
    return signal_f1,all_f1