
import sys  
sys.path.append(sys.path[0].replace('models', ''))

import re
import logging
import math
import json
import pathlib
import numpy as np
from copy import deepcopy
from pathlib import Path
from einops import rearrange
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union, Callable, Optional


import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint

from transformers import AutoModel,BertConfig,AutoTokenizer

from models.transformer_decoder import *

from torch.autograd import Function
import timm
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Tuple, Union
import numpy as np

class MLP_Head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

class Text_Encoder_Bert(nn.Module):
    def __init__(self,
                bert_model_name: str,
                embed_dim: int = 768,
                freeze_layers:Union[Tuple[int, int], int] = None):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=None):#12
        try:
            print(bert_model_name)
            config = AutoConfig.from_pretrained(bert_model_name, output_hidden_states=True)#bert-base-uncased
            print("Config loaded successfully.", flush=True)
            model = AutoModel.from_pretrained(bert_model_name, config=config)#, return_dict=True)
            print("Model loaded successfully.", flush=True)
            print("Text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def encode_text(self, text):
        output = self.bert_model(input_ids = text['input_ids'],attention_mask = text['attention_mask'] )
        last_hidden_state = output.last_hidden_state
        cls_token_output = last_hidden_state[:, 0, :]
        encode_out = self.mlp_embed(cls_token_output)

        return encode_out, last_hidden_state
    
    def forward(self, text):
        output = self.bert_model(input_ids = text['input_ids'],attention_mask = text['attention_mask'] )
        last_hidden_state = output.last_hidden_state
        cls_token_output = last_hidden_state[:, 0, :] 
        encode_out = self.mlp_embed(cls_token_output)  
        return encode_out, last_hidden_state
    
class ModelRes(nn.Module):
    def __init__(self, res_base_model, embed_dim):
        super(ModelRes, self).__init__()
        self.resnet_dict = {
                            "resnet50": models.resnet50(pretrained=True),
                            # "resnet101": models.resnet101(pretrained=True),
                            # "resnet152": models.resnet152(pretrained=True),
                            }
        self.resnet = self._get_res_basemodel(res_base_model)

        num_ftrs = int(self.resnet.fc.in_features)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, embed_dim)

    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, img):
        batch_size = img.shape[0]
        res_fea = self.res_features(img)
        res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool

class ModelRes_3D(nn.Module):
    def __init__(self, config):
        super(ModelRes_3D, self).__init__()
        print('Using Resnet 3D as image encoder!')
        self.resnet = self._get_res_base_model(config['model_type'], config['model_depth'], 
                                               config['input_W'], config['input_H'], config['input_D'],
                                               config['resnet_shortcut'], config['no_cuda'], 
                                               config['gpu_id'], config['pretrain_path'], config['out_feature'])
        
        num_ftrs = int(self.resnet.conv_seg[2].in_features)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-1])
        out_feature = config['out_feature']
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, out_feature)

    def _get_res_base_model(self, model_type, model_depth, input_W, input_H, input_D, resnet_shortcut, no_cuda, gpu_id, pretrain_path, out_feature):
        from models.resnet_3d import resnet50
        if model_depth == 50:
            model = resnet50(sample_input_W=input_W, sample_input_H=input_H, sample_input_D=input_D,
                             shortcut_type=resnet_shortcut, no_cuda=no_cuda, num_seg_classes=1)
            fc_input = 2048

        model.conv_seg = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=fc_input, out_features=out_feature, bias=True)
        )

        net_dict = model.state_dict()
        model = model.cuda()

        if pretrain_path != '':
            print('Loading pretrained model {}'.format(pretrain_path))
            pretrain = torch.load(pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)
            print("-------- Pre-train model loaded successfully --------")
        return model

    def forward(self, images):
        img = images.float().cuda()
        batch_size = img.shape[0]
        res_fea = self.res_features(img)  # [batch_size, feature_size, patch_num, patch_num, patch_num]
        res_fea = rearrange(res_fea, 'b d n1 n2 n3 -> b (n1 n2 n3) d')
        h = rearrange(res_fea, 'b n d -> (b n) d')

        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_embed = rearrange(x, '(b n) d -> b n d', b=batch_size)
        out_pool = torch.mean(out_embed, dim=1)

        return out_embed,out_pool

class ModelConvNeXt(nn.Module):
    def __init__(self, convnext_base_model):
        super(ModelConvNeXt, self).__init__()
        self.convnext_dict = {"convnext-tiny": timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=1000),
                              "convnext-base": timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=1000),
                              "convnext-large": timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=1000),
                              "convnext-huge": timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=1000),
                              }
        convnext = self._get_convnext_basemodel(convnext_base_model)
        num_ftrs = int(convnext.head.in_features)
        self.conv_features = nn.Sequential(*list(convnext.children())[:-2])
        print('num_ftrs for ModelConvNeXt=', num_ftrs)
        self.conv_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.conv_l2 = nn.Linear(num_ftrs, 768)
        
        
    def _get_convnext_basemodel(self, convnext_model_name):
        try:
            convnext_model = self.convnext_dict[convnext_model_name]
            print("Image feature extractor:", convnext_model_name)
            return convnext_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: convnext-tiny, convnext-small or convnext-base")

    def forward(self, img):
        batch_size = img.shape[0]
        conv_fea = self.conv_features(img)
        conv_fea = rearrange(conv_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(conv_fea,'b n d -> (b n) d')
        x = self.conv_l1(h)
        x = F.relu(x)
        x = self.conv_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool


class TQN_Model_fusion(nn.Module):
    def __init__(self, 
            embed_dim: int = 768, 
            class_num: int = 1, 
            lam: list = [1, 0]
            ):
        super().__init__()
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        decoder_layer = TransformerDecoderLayer(self.d_model, 4, 1024,
                                        0.1, 'relu',normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, 4, self.decoder_norm,
                                return_intermediate=False)
        
        self.dropout_feas = nn.Dropout(0.1)

        # class_num = 2
        self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
            nn.Linear(embed_dim, class_num)
        )
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(self, label_features, fusion_features, return_atten = False):
        batch_size = fusion_features.shape[0]

        label_sequence_length = label_features.size(1)

        if batch_size % label_sequence_length != 0:
            repeat_times = (batch_size + label_sequence_length - 1) // label_sequence_length
            label_features = label_features.repeat(1, repeat_times, 1)
            label_features = label_features[:, :batch_size, :]
        else:
            repeat_times = batch_size // label_sequence_length
            label_features = label_features.repeat(1, repeat_times, 1)

        fusion_features = fusion_features.transpose(0,1)

        fusion_features = self.decoder_norm(fusion_features)
        label_features = self.decoder_norm(label_features)

        features,atten_map = self.decoder(label_features, fusion_features, fusion_features, 
                memory_key_padding_mask=None, pos=None, query_pos=None)
                 
        features = self.dropout_feas(features).transpose(0,1)  #b,embed_dim
        out = self.mlp_head(features)  #(batch_size, query_num)

        if return_atten:
            return out, atten_map
        else:
            return out
