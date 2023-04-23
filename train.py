import os
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import numpy as np
from my_dataset import MyDataSet
from model import dual_stream as create_model
from utils import generate_txt_path, train_one_epoch, evaluate, calculate_acc,confusion_matrix,calculate_prediction,calculate_recall,calculate_f1,setup_seed
from d2l import torch as d2l
import scipy.io as scio
def main(args):
    devices=[d2l.try_gpu(i) for i in range(args.GPU_nums)]
    all_matrix=[]
    for round in range(2,25):
        """
        LOSO on MFED,the numbering starts from 2 and ends at 25. Number 5 does not exist.
        """
        if round == 5:
            continue
        setup_seed(11)
        best_accuracy=0.0
        best_epoch=-1
        best_matrix=['模型预测全部错误']
        train_adj = scio.loadmat(os.path.join('.', 'adj', 'train', str(round) + '.mat'))
        train_adj = train_adj['data']
        train_adj = torch.tensor(train_adj)
        train_adj = train_adj.to(torch.float32)
        mask_adj = scio.loadmat(os.path.join('.', 'mask.mat'))
        mask_adj = mask_adj['data']
        mask_adj = torch.tensor(mask_adj)
        mask_adj = mask_adj.to(torch.float32).to(devices[0])
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.Grayscale(),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         ]),
            "val": transforms.Compose([transforms.Grayscale(),
                                       transforms.ToTensor(),
                                       ])}
        train_txt_path,test_txt_path=generate_txt_path(round)
        train_dataset = MyDataSet(train_txt_path,
                                  transform=data_transform["train"])
        val_dataset = MyDataSet(test_txt_path,
                                transform=data_transform["val"])
        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=4,
                                                   collate_fn=train_dataset.collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=4,
                                                 collate_fn=val_dataset.collate_fn)

        model = create_model(num_classes=36, has_logits=False,batchsize=batch_size,A=train_adj,mask=mask_adj)
        model=torch.nn.DataParallel(model,device_ids=devices)

        gcn_parameters_name=['module.gcn.A',
                        'module.gcn.gc1.weight',
                        'module.gcn.gc1.bias',
                        'module.gcn.gc2.weight',
                        'module.gcn.gc2.bias']
        gcn_pg=[]
        pg=[]
        for p in model.named_parameters():
            name,pa=p
            if name in gcn_parameters_name:
                if pa.requires_grad:
                    gcn_pg.append(pa)
            else:
                if pa.requires_grad:
                    pg.append(pa)
        optimizer1 = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
        optimizer2 = optim.Adam(gcn_pg, lr=0.00025)


        for epoch in range(args.epochs):
            if epoch==21:
                for i in gcn_pg:
                    i.requires_grad_(False)

            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer1=optimizer1,
                                                    optimizer2=optimizer2,
                                                    data_loader=train_loader,
                                                    device=devices[0],
                                                    epoch=epoch,
                                                    round=round)


            val_loss, val_acc,matirx,pre,lab= evaluate(model=model,
                                         data_loader=val_loader,
                                         device=devices[0],
                                         epoch=epoch,
                                         round=round)
            if val_acc>best_accuracy:
                best_accuracy=val_acc
                best_model=model.state_dict()
                # torch.save(best_model, "M:/model/model-{}.pth".format(round))
                best_epoch=epoch
                best_matrix=matirx
            if epoch==args.epochs-1:
                # conf_matrix = [[0 for j in range(36)] for i in range(36)]
                conf_matrix = torch.zeros(36, 36, dtype=torch.int64)
                if not os.path.exists(os.path.join('.', 'results')):
                    os.makedirs(os.path.join('.', 'results'))
                with open(os.path.join('.','results',str(round)+'.txt'),'a') as f:
                    f.write('best epoach: '+str(best_epoch)+'\n'+'best acc: '+str(best_accuracy)+'\n'+'matrix: '+str(best_matrix)+'\n')
                if best_matrix!=['模型预测全部错误']:
                    conf_matrix = confusion_matrix(best_matrix, conf_matrix)
                    label_pre,all_pre=calculate_prediction(np.array(conf_matrix))
                    label_recall,all_recall=calculate_recall(np.array(conf_matrix))
                    label_f1,all_f1=calculate_f1(label_pre,all_pre,label_recall,all_recall)
                    with open(os.path.join('.', 'results', str(round) + '.txt'), 'a') as f:
                        f.write('总精确度:'+str(all_pre)+'\n'+'每类精确度:'+str(label_pre)+'\n'+'总召回率:'+str(all_recall)+'\n'+'每类召回率:'+str(label_recall)+'\n'+
                                '总f1:'+str(all_f1)+'\n'+'每类f1:'+str(label_f1)+'\n')
                    all_matrix = all_matrix + best_matrix

    num_right,num_sample,all_acc=calculate_acc(all_matrix)
    all_conf_matrix = torch.zeros(36, 36, dtype=torch.int64)
    all_conf_matrix=confusion_matrix(all_matrix,all_conf_matrix)
    with open(os.path.join('.','results','all.txt'),'a') as f1:
        f1.write('num_right: '+str(num_right)+'\n'+' num_sample: '+str(num_sample)+'\n'+' acc:'+str(all_acc)+'\n')
    all_label_pre, all_all_pre = calculate_prediction(np.array(all_conf_matrix))
    all_label_recall, all_all_recall = calculate_recall(np.array(all_conf_matrix))
    all_label_f1, all_all_f1 = calculate_f1(all_label_pre, all_all_pre, all_label_recall, all_all_recall)
    with open(os.path.join('.', 'results', 'all.txt'), 'a') as f:
        f.write('总精确度:' + str(all_all_pre) + '\n' + '每类精确度:' + str(all_label_pre) + '\n' + '总召回率:' + str(
            all_all_recall) + '\n' + '每类召回率:' + str(all_label_recall) + '\n' +
                '总f1:' + str(all_all_f1) + '\n' + '每类f1:' + str(all_label_f1) + '\n')


            # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str,
                        default="./data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--rounds',type=int,default=24)
    parser.add_argument('--GPU_nums',type=int,default=1)
    opt = parser.parse_args()
    main(opt)
