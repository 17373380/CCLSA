"""
Name: train_process
Date: 2022/4/11 上午10:26
Version: 1.0

"""

import torch
# from transformers import AdamW
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from util.write_file import WriteFile
import dev_process
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam
# import tensorflow as tf


def train_process(opt, train_loader, dev_loader, test_loader, cl_model, critertion, log_summary_writer:SummaryWriter=None, tokenizer=None, image_id_list=None):
    optimizer = None

    pre_train_model_param = [name for name, param in cl_model.named_parameters() if 'text_model' in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in cl_model.named_parameters() if n in pre_train_model_param],
            "lr": 0,
        },
        {
            "params": [p for n, p in cl_model.named_parameters() if n not in pre_train_model_param],
            "lr": opt.fuse_lr,
        },
    ]

    #优化器默认adam
    if opt.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum)

    orgin_param = ModelParam()
    augment_param = ModelParam()
    augment_param_2 = ModelParam()

    last_F1 = 0
    last_Accuracy = 0


    #训练开始
    for epoch in trange(opt.epoch, desc='Epoch:'):
        y_true = []
        y_pre = []
        run_loss = 0
        total_labels = 0

        cl_model.train()
        cl_model.zero_grad()

        #如果当前轮次是只训练融合层的轮次 默认为10
        if epoch >= opt.train_fuse_model_epoch:
            #调整优化器的参数
            optimizer.param_groups[0]['lr'] = opt.lr
            optimizer.param_groups[1]['lr'] = opt.lr

        # 创建一个带有进度条的加载器
        train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
        epoch_step_num = epoch * train_loader_tqdm.total
        step_num = 0
        # 把数据按照每个小批量进行取出
        for index, data in enumerate(train_loader_tqdm):
            #print(data)
            texts_origin, bert_attention_mask, image_origin, text_image_mask, labels,\
                texts_augment, bert_attention_mask_augment, texts_caption, caption_mask, image_augment, \
                text_image_mask_augment, caption_image_mask, text_caption_mask, target_labels = data
            # print(labels)

            #设置cuda
            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
                texts_augment = texts_augment.cuda()
                bert_attention_mask_augment = bert_attention_mask_augment.cuda()
                caption_mask = caption_mask.cuda()
                texts_caption = texts_caption.cuda()
                image_augment = image_augment.cuda()
                text_image_mask_augment = text_image_mask_augment.cuda()
                caption_image_mask = caption_image_mask.cuda()
                text_caption_mask = text_caption_mask.cuda()
                for i in range(len(target_labels)):
                    target_labels[i] = target_labels[i].cuda()

            # 原始数据
            orgin_param.set_data_param(texts=texts_origin, bert_attention_mask=bert_attention_mask, images=image_origin,
                                        text_image_mask=text_image_mask)
            # 增强数据
            augment_param.set_data_param(texts=texts_caption, bert_attention_mask=caption_mask, images=image_augment,
                                         text_image_mask=caption_image_mask, text_caption_mask=text_caption_mask)
            augment_param_2.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment)

            # 三维的输出，原始数据和增强数据的张量，一个大小和l_pos_neg.size(0)一样的整数序列，得到原始数据的损失
            origin_res, l_pos_neg, cl_lables, cl_self_loss, l_cap_neg = cl_model(orgin_param, augment_param,
                                                                                 augment_param_2, labels, target_labels)
            #print(l_cap_neg)

            # 这个是基于标签的对比学习，critertion 是一个损失函数在这里为交叉熵损失，origin_res是模型的输出预测结果，labels是真实标签，计算出差异并保存在classify_loss中
            # 基于标签的损失在cl_self_loss中

            # 这个是主要损失
            classify_loss = critertion(origin_res, labels)
            # 这个是基于增强数据的对比学习，同理计算交叉熵损失
            cl_loss = critertion(l_pos_neg, cl_lables)

            # 新加入cap_los
            cap_los = critertion(l_cap_neg, cl_lables)

            # 损失函数包含三重损失
            # loss = (classify_loss + cl_loss * opt.cl_loss_alpha + cl_self_loss * opt.cl_self_loss_alpha) / opt.acc_batch_size
            # 这个是我的损失
            loss = (classify_loss + cl_loss * opt.cl_loss_alpha + cap_los * opt.cap_loss_alpha) / opt.acc_batch_size
            #这个是什么都不要的损失
            #loss = classify_loss / opt.acc_batch_size
            #这个是caption+image对比学习的损失
            # loss = (classify_loss + cl_loss * opt.cl_loss_alpha) / opt.acc_batch_size
            #这个是text+caption对比学习的损失
            # loss = (classify_loss + cap_los * opt.cap_loss_alpha) / opt.acc_batch_size
            loss.backward()  # 反向传播
            train_loader_tqdm.set_description("Train Iteration, loss: %.6f, lr: %e" %
                                              (loss, optimizer.param_groups[0]['lr']))

            if (index + 1) % opt.acc_grad == 0:
                if log_summary_writer:
                    log_summary_writer.add_scalar('train_info/loss', loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/classify_loss', classify_loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/cl_loss', cl_loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/cl_self_loss', cl_self_loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/lr', optimizer.param_groups[0]['lr'], global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/fuse_lr', optimizer.param_groups[1]['lr'], global_step=step_num + epoch_step_num)
                optimizer.step()
                optimizer.zero_grad()
            step_num += 1

            _, predicted = torch.max(origin_res, 1)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            run_loss += loss.item()
            total_labels += labels.size(0)

        run_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        train_accuracy = accuracy_score(y_true, y_pre)
        train_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        train_R_weighted = recall_score(y_true, y_pre, average='weighted')
        train_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        train_F1 = f1_score(y_true, y_pre, average='macro')
        train_R = recall_score(y_true, y_pre, average='macro')
        train_precision = precision_score(y_true, y_pre, average='macro')

        save_content = 'Epoch: %d:\nTrain: Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
                       (epoch, train_accuracy, train_F1_weighted, train_precision_weighted, train_R_weighted, train_F1, train_precision, train_R, run_loss)
        WriteFile(opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
        print(save_content, ' ' * 200)

        if log_summary_writer:
            log_summary_writer.add_scalar('train_info/loss_epoch', run_loss, global_step=epoch)
            log_summary_writer.add_scalar('train_info/acc', train_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_w', train_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/r_w', train_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/p_w', train_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_ma', train_F1, global_step=epoch)
            log_summary_writer.flush()

        train_log = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_F1": train_F1,
            "train_R": train_R,
            "train_precision": train_precision,
            "train_F1_weighted": train_F1_weighted,
            "train_precision_weighted": train_precision_weighted,
            "train_R_weighted": train_R_weighted,
            "run_loss": run_loss
        }
        # debug：正常运行不要把下面的代码注释掉
        last_F1, last_Accuracy = dev_process.dev_process(opt, critertion, cl_model, dev_loader, test_loader, last_F1, last_Accuracy, train_log, log_summary_writer)
