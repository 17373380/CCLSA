"""
Name: model
Date: 2022/4/11 上午10:25
Version: 1.0
"""

import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
import os
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, \
    AlbertConfig, RobertaTokenizer
import math
import matplotlib.pyplot as plt
from pre_model import RobertaEncoder
import copy
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import json
# 测试添加部分

# 可视化 Grad-CAM
def visualize_grad_cam(grad_cam, input_image):
    # 将 Grad-CAM 映射到输入图像上
    # 这部分的具体实现取决于你的数据和图像预处理方式
    # 可以将热力图叠加到图像上或者进行其他可视化处理

    # 以下是示例代码，展示如何将 Grad-CAM 映射到图像上
    plt.figure()
    plt.imshow(input_image, cmap='gray')  # 原始图像
    plt.imshow(grad_cam, cmap='jet', alpha=0.5)  # Grad-CAM 热力图
    plt.colorbar()
    plt.title("Grad-CAM")
    plt.show()

# 测试添加部分

class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None,
                 image_coordinate_position_token=None, text_caption_mask=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.text_caption_mask = text_caption_mask

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None,
                       segment_token=None, image_coordinate_position_token=None, text_caption_mask=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.text_caption_mask = text_caption_mask


def get_extended_attention_mask(attention_mask, input_shape):  # 进行维度扩展
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class BertClassify(nn.Module):
    def __init__(self, opt, in_feature, dropout_rate=0.1):
        super(BertClassify, self).__init__()
        self.classify_linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_feature, 3),
            ActivateFun(opt)
        )

    def forward(self, inputs):
        return self.classify_linear(inputs)


class TextModel(nn.Module):  # 文本模型
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = ''

        if opt.text_model == 'bert-base':
            self.config = BertConfig.from_pretrained(abl_path + 'bert-base-uncased/')  # bert 配置路径
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert-base-uncased/', config=self.config)  # 模型路径
            self.model = self.model.bert
        elif opt.text_model == 'roberta-base':
            self.config = RobertaConfig.from_pretrained(abl_path + 'roberta-base/')
            self.model = RobertaModel.from_oretrained(abl_path + 'roberta-base/', config=self.config)
            self.model = self.model.roberta

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):  # 模型输出
        output = self.model(input, attention_mask=attention_mask)
        return output


class ImageModel(nn.Module):  # 选择图像处理模型
    def __init__(self, opt):
        super(ImageModel, self).__init__()  # 默认使用ResNet50
        if opt.image_model == 'resnet-152':
            self.resnet = cv_models.resnet152(pretrained=True)
        elif opt.image_model == 'resnet-101':
            self.resnet = cv_models.resnet101(pretrained=True)
        elif opt.image_model == 'resnet-50':
            self.resnet = cv_models.resnet50(pretrained=True)
        elif opt.image_model == 'resnet-34':
            self.resnet = cv_models.resnet34(pretrained=True)
        elif opt.image_model == 'resnet-18':
            self.resnet = cv_models.resnet18(pretrained=True)
        self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))  # 把resnet最后两层给删除
        self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])  # avgpool层提取出来
        self.output_dim = self.resnet_encoder[7][2].conv3.out_channels  # 拿出输出层
        #

        #

        for param in self.resnet.parameters():  # 是否固定图像模型的参数
            if opt.fixed_image_model:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        image_encoder = self.resnet_encoder(images)  # 输入图像

        #
        '''
        self.target_layer = nn.Sequential(*(list(self.resnet.children())[-3]))
        cam = GradCAM(model=self.resnet, target_layers=self.target_layer, use_cuda=False)
        target_category = 0
        grayscale_cam = cam(input_tensor=image_encoder, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]

        img_path = "/home/yaobowen/CLMLF-main/dataset/data/MVSA-single/dataset_image/2.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                          grayscale_cam,
                                          use_rgb=True)
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            class_indict = json.load(f)
        plt.title(class_indict[str(target_category)])
        plt.imshow(visualization)
        plt.show()
        '''


        #

        # image_encoder = self.conv_output(image_encoder)
        image_cls = self.resnet_avgpool(image_encoder)  # 平均池化
        image_cls = torch.flatten(image_cls, 1)  # 将图像特征从第二个维度开始展平，其余的维度不变。
        return image_encoder, image_cls


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.fuse_type = opt.fuse_type  # att, ave, max
        self.image_output_type = opt.image_output_type
        self.zoom_value = math.sqrt(opt.tran_dim)
        self.save_image_index = 0

        # 得到使用的文字和图像模型
        self.text_model = TextModel(opt)
        self.image_model = ImageModel(opt)

        self.text_config = copy.deepcopy(self.text_model.get_config())
        self.image_config = copy.deepcopy(self.text_model.get_config())

        self.text_config.num_attention_heads = opt.tran_dim // 64
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers

        self.image_config.num_attention_heads = opt.tran_dim // 64
        self.image_config.hidden_size = opt.tran_dim
        self.image_config.num_hidden_layers = opt.image_num_layers

        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)
        self.image_encoder = RobertaEncoder(self.image_config)

        self.bilstm = nn.LSTM(opt.tran_dim, 384, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        # 将特征传递给全连接层降维，并用激活函数进行非线性变换
        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),  # 将文本特征从文本编码的输出维度降到目标维度
            ActivateFun(opt)  # 激活函数的选择
        )
        self.image_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),  # 将图像特征从图像编码的输出维度降到目标维度
            ActivateFun(opt)  # 激活函数
        )
        self.image_cls_change = nn.Sequential(  # 进行平滑变化
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )

        # 创建一个标准化层和一个dropout层
        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        # transformer设置
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim // 64,
                                                               dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                         num_layers=opt.tran_num_layers)

        # 注意力模块
        if self.fuse_type == 'att':
            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )

        # 分类函数
        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 3)
        )

    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask):
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)  # 得到文本特征
        #print(type(text_encoder))
        text_cls = text_encoder.pooler_output  #
        text_encoder = text_encoder.last_hidden_state  # 最后一个文本隐藏层的状态
        #print(text_encoder.size())
        text_init = self.text_change(text_encoder)  # 通过全连接层和激活函数
        #print(text_init.size())
        text_init, (hidden_state, cell_state) = self.bilstm(text_init)
        text_init = self.dropout(text_init)
        #print(text_init.size())

        image_encoder, image_cls = self.image_model(image_inputs)  # 得到从resnet输出的特征以及平滑变化后的特征
        # cls是平滑后的图像特征
        if self.image_output_type == 'all':
            # 改变图像特征的结构
            image_encoder = image_encoder.contiguous().view(image_encoder.size(0), -1, image_encoder.size(1))
            image_encoder_init = self.image_change(image_encoder)  # 改变后的特征通过全连接层和激活函数
            image_cls_init = self.image_cls_change(image_cls)  # 平滑后的特征通过全连接层和激活函数
            image_init = torch.cat((image_cls_init.unsqueeze(1), image_encoder_init), dim=1)  # 将图像原始特征和图像平滑特征进行在维度1上进行拼接
        else:
            image_cls_init = self.image_cls_change(image_cls)  # 平滑后的特征通过全连接层和激活函数
            image_init = image_cls_init.unsqueeze(1)  # 特征扩展

        # 创建掩码
        image_mask = text_image_mask[:, -image_init.size(1):]
        # 掩码维度扩展
        extended_attention_mask = get_extended_attention_mask(image_mask, image_init.size())

        # self.image_encoder = RobertaEncoder(self.image_config)
        # 图像部分先通过一次transformer层
        image_init = self.image_encoder(image_init,
                                        attention_mask=None,
                                        head_mask=None,
                                        encoder_hidden_states=None,
                                        encoder_attention_mask=extended_attention_mask,
                                        past_key_values=None,
                                        use_cache=self.use_cache,
                                        output_attentions=self.text_config.output_attentions,
                                        output_hidden_states=(self.text_config.output_hidden_states),
                                        return_dict=self.text_config.use_return_dict
                                        )
        image_init = image_init.last_hidden_state  # 最后一层隐藏层的结果，为图像特征

        # 添加

        # 添加

        # 文本和图像信息拼接
        text_image_cat = torch.cat((text_init, image_init), dim=1)

        # 对特征进行扩展
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_image_mask, text_inputs.size())
        #self.text_image_encoder = RobertaEncoder(self.text_config)
        # 经过拼接后的特征再次输入到transformer中
        text_image_transformer = self.text_image_encoder(text_image_cat,
                                                         attention_mask=extended_attention_mask,
                                                         head_mask=None,
                                                         encoder_hidden_states=None,
                                                         encoder_attention_mask=extended_attention_mask,
                                                         past_key_values=None,
                                                         use_cache=self.use_cache,
                                                         output_attentions=self.text_config.output_attentions,
                                                         output_hidden_states=(self.text_config.output_hidden_states),
                                                         return_dict=self.text_config.use_return_dict)
        text_image_transformer = text_image_transformer.last_hidden_state  # 提取出最后影藏层的特征
        text_image_transformer = text_image_transformer.permute(0, 2, 1).contiguous()  # 将张量的第一维和第二维进行互换

        #fuse_type为最大值
        if self.fuse_type == 'max':
            text_image_output = torch.max(text_image_transformer, dim=2)[0]  # 得到维度2中包含最大值的张量
        elif self.fuse_type == 'att':
            text_image_output = text_image_transformer.permute(0, 2, 1).contiguous()

            text_image_mask = text_image_mask.permute(1, 0).contiguous()
            text_image_mask = text_image_mask[0:text_image_output.size(1)]
            text_image_mask = text_image_mask.permute(1, 0).contiguous()

            text_image_alpha = self.output_attention(text_image_output)
            text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_image_mask == 0, -1e9)
            text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

            text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)
        elif self.fuse_type == 'ave':
            text_image_length = text_image_transformer.size(2)
            text_image_output = torch.sum(text_image_transformer, dim=2) / text_image_length
        else:
            raise Exception('fuse_type设定错误')
        #print(text_image_output)
        return text_image_output, None, None


class CaptionModel(nn.Module):
    def __init__(self, opt):
        super(CaptionModel, self).__init__()
        self.fuse_type = opt.fuse_type  # att, ave, max
        self.save_image_index = 0

        # 得到使用的文字和图像模型
        self.text_model = TextModel(opt)

        self.text_config = copy.deepcopy(self.text_model.get_config())

        self.text_config.num_attention_heads = opt.tran_dim // 64
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers

        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)

        self.bilstm = nn.LSTM(opt.tran_dim, 384, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        # 将特征传递给全连接层降维，并用激活函数进行非线性变换
        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),  # 将文本特征从文本编码的输出维度降到目标维度
            ActivateFun(opt)  # 激活函数的选择
        )

        # 创建一个标准化层和一个dropout层
        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        # transformer设置
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim // 64,
                                                               dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                         num_layers=opt.tran_num_layers)

        # 注意力模块
        if self.fuse_type == 'att':
            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )

        # 分类函数
        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 3)
        )

    def forward(self, text_inputs, bert_attention_mask, caption_inputs, caption_mask, text_caption_mask):
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)  # 得到文本特征
        text_cls = text_encoder.pooler_output  #
        text_encoder = text_encoder.last_hidden_state  # 最后一个文本隐藏层的状态
        text_init = self.text_change(text_encoder)  # 通过全连接层和激活函数
        text_init, (hidden_state, cell_state) = self.bilstm(text_init)
        text_init = self.dropout(text_init)

        caption_encoder = self.text_model(caption_inputs, attention_mask=caption_mask)
        caption_cls = caption_encoder.pooler_output
        caption_encoder = caption_encoder.last_hidden_state
        caption_init = self.text_change(caption_encoder)
        caption_init, (hidden_state, cell_state) = self.bilstm(caption_init)
        caption_init = self.dropout(caption_init)

        # 文本和图像信息拼接
        text_caption_cat = torch.cat((text_init, caption_init), dim=1)

        # 对特征进行扩展
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_caption_mask, text_inputs.size())

        # self.text_image_encoder = RobertaEncoder(self.text_config)
        # 经过拼接后的特征再次输入到transformer中
        text_caption_transformer = self.text_image_encoder(text_caption_cat,
                                                           attention_mask=extended_attention_mask,
                                                           head_mask=None,
                                                           encoder_hidden_states=None,
                                                           encoder_attention_mask=extended_attention_mask,
                                                           past_key_values=None,
                                                           use_cache=self.use_cache,
                                                           output_attentions=self.text_config.output_attentions,
                                                           output_hidden_states=(
                                                               self.text_config.output_hidden_states),
                                                           return_dict=self.text_config.use_return_dict)
        text_caption_transformer = text_caption_transformer.last_hidden_state  # 提取出最后影藏层的特征
        text_caption_transformer = text_caption_transformer.permute(0, 2, 1).contiguous()  # 将张量的第一维和第二维进行互换

        # fuse_type为att
        if self.fuse_type == 'max':
            text_caption_output = torch.max(text_caption_transformer, dim=2)[0]  # 得到维度2中包含最大值的张量
        elif self.fuse_type == 'att':
            text_caption_output = text_caption_transformer.permute(0, 2, 1).contiguous()

            text_caption_mask = text_caption_mask.permute(1, 0).contiguous()
            text_caption_mask = text_caption_mask[0:text_caption_output.size(1)]
            text_caption_mask = text_caption_mask.permute(1, 0).contiguous()

            text_caption_alpha = self.output_attention(text_caption_output)
            text_caption_alpha = text_caption_alpha.squeeze(-1).masked_fill(text_caption_mask == 0, -1e9)
            text_caption_alpha = torch.softmax(text_caption_alpha, dim=-1)

            text_caption_output = (text_caption_alpha.unsqueeze(-1) * text_caption_output).sum(dim=1)
        elif self.fuse_type == 'ave':
            text_caption_length = text_caption_transformer.size(2)
            text_caption_output = torch.sum(text_caption_transformer, dim=2) / text_caption_length
        else:
            raise Exception('fuse_type设定错误')
        # print(text_image_output)
        return text_caption_output, None, None


class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(opt)  # 加载fuse模型
        self.caption_model = CaptionModel(opt)
        self.temperature = opt.temperature  # 用于计算对比学习损失
        self.set_cuda = opt.cuda

        # 线性层加激活函数加线性层
        self.orgin_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        # 线性层加激活函数加线性层
        self.augment_linear_change = nn.Sequential(
            nn.Linear(opt.tran_dim, opt.tran_dim),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim, opt.tran_dim)
        )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),  # dropout层，随机丢弃元素，减少过拟合。
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),  # 缩减维度
            ActivateFun(opt),  # 激活函数
            nn.Linear(opt.tran_dim // 2, 3)  # 将维度变换为3
        )

    def forward(self, data_orgin: ModelParam, data_augment: ModelParam = None, data_augment_2: ModelParam = None, labels=None, target_labels=None):
        # 得到图像和文本拼接过后的特征
        orgin_res, orgin_text_cls, orgin_image_cls = self.fuse_model(data_orgin.texts, data_orgin.bert_attention_mask,
                                                                     data_orgin.images, data_orgin.text_image_mask)
        #print(orgin_res.size())
        # 将维度分类成三类
        output = self.output_classify(orgin_res)
        # print(output.size())

        # 数据增强后的数据
        if data_augment:
            augment_res, augment_text_cls, augment_image_cls = self.fuse_model(data_augment.texts,
                                                                               data_augment.bert_attention_mask,
                                                                               data_augment.images,
                                                                               data_augment.text_image_mask)

            caption_res, augment_text_cls, augment_caption_cls = self.caption_model(data_augment_2.texts,
                                                                                    data_augment_2.bert_attention_mask,
                                                                                    data_augment.texts,
                                                                                    data_augment.bert_attention_mask,
                                                                                    data_augment.text_caption_mask)

            orgin_res_change = self.orgin_linear_change(orgin_res)  # 普通数据进行变换
            augment_res_change = self.augment_linear_change(augment_res)  # 增强数据进行变换

            # 得到两个张量之间的内积
            l_pos_neg = torch.einsum('nc,ck->nk', [orgin_res_change, augment_res_change.T])
            # 创建一个大小和l_pos_neg.size(0)一样的整数序列
            cl_lables = torch.arange(l_pos_neg.size(0))
            if self.set_cuda:
                cl_lables = cl_lables.cuda()
            l_pos_neg /= self.temperature  # 将内积中的每个元素都除以温度值

            #修改部分
            caption_res_change = self.augment_linear_change(caption_res)
            l_cap_neg = torch.einsum('nc,ck->nk', [orgin_res_change, caption_res_change.T])
            cl_cap_lables = torch.arange(l_cap_neg.size(0))
            if self.set_cuda:
                cl_cap_lables = cl_cap_lables.cuda()
            l_cap_neg /= self.temperature  # 将内积中的每个元素都除以温度值
            #修改部分完成

            # 计算普通数据的内积
            l_pos_neg_self = torch.einsum('nc,ck->nk', [orgin_res_change, orgin_res_change.T])
            l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1)  # 在l_pos_neg_self列维度上进行softmax操作
            l_pos_neg_self = l_pos_neg_self.view(-1)  # 将张量转化成一维的，并且不改变顺序

            cl_self_labels = target_labels[labels[0]]  # 取出第一个子集
            for index in range(1, orgin_res.size(0)):
                # 遍历orgin_res张量的多个元素，根据每个元素的值从target_labels中选择相应的子集，然后将这些子集连接起来形成一个新的张量cl_self_labels
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[index]] + index * labels.size(0)), 0)

            # 将每个值除以温度
            l_pos_neg_self = l_pos_neg_self / self.temperature
            # 根据索引值按照行向量对内积进行搜索数据
            cl_self_loss = torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels)
            cl_self_loss = - cl_self_loss.sum() / cl_self_labels.size(0)
            # 综合起来，这行代码的功能是计算 cl_self_loss 张量中的负对数似然损失，并求取其平均值。
            # 负对数似然损失越小，表示模型的预测越接近真实标签，损失越大表示预测偏离真实标签。具体的用途和上下文可能需要查看代码的其余部分来确定。

            return output, l_pos_neg, cl_lables, cl_self_loss, l_cap_neg
        else:
            return output


class TensorBoardModel(nn.Module):
    def __init__(self, opt):
        super(TensorBoardModel, self).__init__()
        self.cl_model = CLModel(opt)

    def forward(self, texts, bert_attention_mask, images, text_image_mask,
                texts_augment, bert_attention_mask_augment, images_augment, text_image_mask_augment, label):
        orgin_param = ModelParam()
        augment_param = ModelParam()
        orgin_param.set_data_param(texts=texts, bert_attention_mask=bert_attention_mask, images=images,
                                   text_image_mask=text_image_mask)
        augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment,
                                     images=images_augment, text_image_mask=text_image_mask_augment)
        return self.cl_model(orgin_param, augment_param, label, [torch.ones(1, dtype=torch.int64) for _ in range(3)])
