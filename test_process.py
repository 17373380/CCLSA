"""
Name: test_process
Date: 2022/4/11 上午10:26
Version: 1.0
"""

from model import ModelParam
import torch
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import numpy as np
from sklearn import manifold, datasets
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('AGG') # pylint: disable=multiple-statements
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
# import tensorflow as tf
import math


def test_process(opt, critertion, cl_model, test_loader, last_F1=None, log_summary_writer: SummaryWriter=None, epoch=None):
    y_true = []
    y_pre = []
    total_labels = 0
    test_loss = 0

    orgin_param = ModelParam()

    with torch.no_grad():
        cl_model.eval()
        test_loader_tqdm = tqdm(test_loader, desc='Test Iteration')
        epoch_step_num = epoch * test_loader_tqdm.total
        step_num = 0

        #
        fig = plt.figure(figsize=(8, 8))  # 指定图像的宽和高
        plt.suptitle("Dimensionality Reduction and Visualization of S-Curve Data ", fontsize=14)
        ax1 = fig.add_subplot(1, 1, 1)
        tensor_list = []
        labels_list = []
        #
        for index, data in enumerate(test_loader_tqdm):
            texts_origin, bert_attention_mask, image_origin, text_image_mask, labels, \
            texts_augment, bert_attention_mask_augment, texts_caption, caption_mask, image_augment, caption_text_image_mask, text_image_mask_augment, text_caption_mask, _ = data
            # continue

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()

            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin,
                                       text_image_mask=text_image_mask)
            origin_res = cl_model(orgin_param)

            tensor_list.append(origin_res)
            labels_list.append(labels)

            '''
            ts = manifold.TSNE(n_components=2, random_state=0)
            #print(origin_res)
            y = ts.fit_transform(origin_res.cpu())
            for index, x in enumerate(y):
                if labels.cpu()[index] == 0:
                    color = 'r'
                elif labels.cpu()[index] == 1:
                    color = 'g'
                else:
                    color = 'b'

                if ((x[0] > -500) and (x[0] < 500)) and ((-500 < x[1]) and (x[1] < 500)):
                    plt.scatter(x[0], x[1], c=color, alpha=0.2)
            '''


            #print(origin_res)

            loss = critertion(origin_res, labels) / opt.acc_batch_size
            test_loss += loss.item()
            _, predicted = torch.max(origin_res, 1)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())

            test_loader_tqdm.set_description("Test Iteration, loss: %.6f" % loss)
            if log_summary_writer:
                log_summary_writer.add_scalar('test_info/loss', loss.item(), global_step=step_num + epoch_step_num)
            step_num += 1

        result_tensor = torch.cat(tensor_list, dim=0)
        res_labels = torch.cat(labels_list)
        #print(res_labels)
        ts = manifold.TSNE(n_components=2, random_state=0)
        # print(origin_res)
        y = ts.fit_transform(result_tensor.cpu())
        #print(y)
        for index, x in enumerate(y):
            if res_labels.cpu()[index] == 0:
                color = 'tomato'
            elif res_labels.cpu()[index] == 1:
                color = 'lightpink'
            else:
                color = 'cornflowerblue'

            plt.scatter(x[0], x[1], c=color, alpha=0.5)
            #if ((x[0] > -500) and (x[0] < 500)) and ((-500 < x[1]) and (x[1] < 500)):

        #print(result_tensor)
        ax1.set_title('t-SNE Curve', fontsize=14)
        plt.savefig("Cluster.png")
        test_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        test_accuracy = accuracy_score(y_true, y_pre)
        test_F1 = f1_score(y_true, y_pre, average='macro')
        test_R = recall_score(y_true, y_pre, average='macro')
        test_precision = precision_score(y_true, y_pre, average='macro')
        test_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        test_R_weighted = recall_score(y_true, y_pre, average='weighted')
        test_precision_weighted = precision_score(y_true, y_pre, average='weighted')

        save_content = 'Test : Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
            (test_accuracy, test_F1_weighted, test_precision_weighted, test_R_weighted, test_F1, test_precision, test_R, test_loss)

        print(save_content)

        if log_summary_writer:
            log_summary_writer.add_scalar('test_info/loss_epoch', test_loss, global_step=epoch)
            log_summary_writer.add_scalar('test_info/acc', test_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('test_info/f1_w', test_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/r_w', test_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/p_w', test_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/f1_ma', test_F1, global_step=epoch)
            log_summary_writer.flush()

        if last_F1 is not None:
            WriteFile(
                opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
