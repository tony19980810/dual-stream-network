import os

import numpy as np
import torch

from utils import calculate_acc, calculate_prediction, calculate_recall, calculate_f1,confusion_matrix


def calculate_result():
    """
    caculate overall acc,F1,recall using txt result of every subject in LOSO
    """
    all_matrix=[]
    for root,dirs,files in os.walk(os.path.join('.','results')):
        for i in files:
            with open(os.path.join('.','results',i), "r") as f:
                data = f.readlines()
                matrix=[]
                tmp=data[2][9:-2]
                print(tmp)
                tmp2=tmp.split(', ')
                tmpnum1 = ''
                tmpnum2 = ''
                for j in range(0,len(tmp2)):
                    if j%2==0:
                        for l in tmp2[j]:
                            if l>='0' and l<='9':
                                tmpnum1=tmpnum1+l
                    else:
                        for l in tmp2[j]:
                            if l>='0' and l<='9':
                                tmpnum2=tmpnum2+l
                        matrix.append([int(tmpnum1),int(tmpnum2)])
                        tmpnum1 = ''
                        tmpnum2 = ''
                all_matrix=all_matrix+matrix
    num_right, num_sample, all_acc = calculate_acc(all_matrix)
    all_conf_matrix = torch.zeros(36, 36, dtype=torch.int64)
    all_conf_matrix = confusion_matrix(all_matrix, all_conf_matrix)
    with open(os.path.join('.', 'results', 'all.txt'), 'a') as f1:
        f1.write('num_right: ' + str(num_right) + '\n' + ' num_sample: ' + str(num_sample) + '\n' + ' acc:' + str(
            all_acc) + '\n')
    all_label_pre, all_all_pre = calculate_prediction(np.array(all_conf_matrix))
    all_label_recall, all_all_recall = calculate_recall(np.array(all_conf_matrix))
    all_label_f1, all_all_f1 = calculate_f1(all_label_pre, all_all_pre, all_label_recall, all_all_recall)
    with open(os.path.join('.', 'results', 'all.txt'), 'a') as f:
        f.write('总精确度:' + str(all_all_pre) + '\n' + '每类精确度:' + str(all_label_pre) + '\n' + '总召回率:' + str(
            all_all_recall) + '\n' + '每类召回率:' + str(all_label_recall) + '\n' +
                '总f1:' + str(all_all_f1) + '\n' + '每类f1:' + str(all_label_f1) + '\n')


calculate_result()

