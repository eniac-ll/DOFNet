import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

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
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    all_preds = []
    all_labels = []
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        current_f1 = f1_score(all_labels, all_preds, average='macro')

        data_loader.desc = "[train epoch {}] train_loss: {:.4f}, train_acc: {:.2f}%  f1: {:.2f}%     ".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               100*accu_num.item() / sample_num,
                                                                                100*current_f1.item())
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    final_f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, final_f1, kappa

from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix,recall_score,precision_score
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    all_preds = []
    all_labels = []

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] valid_loss: {:.4f}, valid_acc: ------>{:.2f}% ".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               100*accu_num.item() / sample_num)
    cm = confusion_matrix(all_labels, all_preds)

    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)


    final_f1 = f1_score(all_labels, all_preds, average='macro')
    final_recall = recall_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,final_f1,kappa,cm,final_recall

