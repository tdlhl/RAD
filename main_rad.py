import argparse
import os
import logging
import yaml
import numpy as np
import random
import time
import datetime
import json
import math
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from transformers import AutoModel,BertConfig,AutoTokenizer

from factory import utils
from scheduler import create_scheduler
from optim import create_optimizer_dual
from engine.train_rad import valid_on_ICD, train_grad_acc, test_logits

from models.clip_tqn import ModelRes, Text_Encoder_Bert, TQN_Model_fusion, ModelConvNeXt, ModelRes_3D
from models.tokenization_bert import BertTokenizer
from dataset.dataset_entity import ICD_Train_Dataset, Fair_ori_train_dataset, Skin_Train_Dataset, NACC_Train_Dataset
from dataset.test_dataset import ICD_Dataset, Fair_ori_test_dataset, Skin_Test_Dataset, NACC_Test_Dataset


import socket
from io import BytesIO


def seed_torch(seed=42):
    print('=====> Using fixed random seed: ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args, config):
    torch.cuda.current_device()
    torch.cuda._initialized = True
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')
    
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)#cuda

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    print('sampler_rank',sampler_rank,'num_tasks',num_tasks)

    #### Dataset #### 
    print("Creating dataset")
    if 'fair_ori' in args.dataset:
        train_dataset = Fair_ori_train_dataset(config['ICD_train_file'], config['image_res'])
    elif 'skin' in args.dataset:
        train_dataset = Skin_Train_Dataset(config['ICD_train_file'], config['image_res'])
    elif 'nacc' in args.dataset:
        train_dataset = NACC_Train_Dataset(config['ICD_train_file'], config['image_res'])
    else:
        train_dataset = ICD_Train_Dataset(config['ICD_train_file'], config['image_res'])
    
  
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=8,
            pin_memory=True,
            sampler=train_sampler, 
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=True,
        )    
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader)  

    if 'fair_ori' in args.dataset:
        val_dataset = Fair_ori_test_dataset(config['ICD_test_file'],config['image_res'])
    elif 'skin' in args.dataset:
        val_dataset = Skin_Test_Dataset(config['ICD_test_file'],config['image_res'])
    elif 'nacc' in args.dataset:
        val_dataset = NACC_Test_Dataset(config['ICD_test_file'], config['image_res'])
    else:
        val_dataset = ICD_Dataset(config['ICD_test_file'],config['image_res'])
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    val_dataloader =DataLoader(
            val_dataset,
            batch_size=config['test_batch_size'],
            num_workers=8,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=False,
        )     
    val_dataloader.num_samples = len(val_dataset)
    val_dataloader.num_batches = len(val_dataloader) 

    if 'res' in args.image_encoder_name:
        if 'nacc' in args.dataset:
            resnet_3d_config = {
                'model_type': 'resnet',
                'model_depth': 50,
                'input_W': 96,
                'input_H': 96,
                'input_D': 96,
                'resnet_shortcut': 'B',
                'no_cuda': False,
                'gpu_id': 0,
                'pretrain_path': '',
                'out_feature': args.embed_dim
            }
            image_encoder = ModelRes_3D(resnet_3d_config).cuda()
        else:
            image_encoder = ModelRes(res_base_model=args.image_encoder_name, embed_dim=args.embed_dim).cuda()
    elif 'convnext' in args.image_encoder_name:
        image_encoder = ModelConvNeXt(convnext_base_model=args.image_encoder_name).cuda()
    else:
        raise ValueError('invalid image encoder', args.image_encoder_name)

    if args.bert_model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name,do_lower_case=True, local_files_only=True)
        text_encoder = Text_Encoder_Bert(bert_model_name=args.bert_model_name).cuda()

    model = TQN_Model_fusion(embed_dim=args.embed_dim).cuda()
    model_guideline = TQN_Model_fusion(embed_dim=args.embed_dim).cuda()
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer_dual(arg_opt, model, model_guideline, image_encoder, text_encoder)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    if os.path.exists(args.output_dir):
        checkpoints = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
            checkpoint_path = os.path.join(args.output_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model.load_state_dict(checkpoint['model'])
            model_guideline.load_state_dict(checkpoint['model_guideline'])
            image_encoder.load_state_dict(checkpoint['image_encoder'])
            text_encoder.load_state_dict(checkpoint['text_encoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}, checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found, starting training from scratch.")
    else:
        print("Output directory does not exist, starting training from scratch.")

    print("Start training")
    start_time = time.time()
    if utils.is_main_process(): 
        writer = SummaryWriter(args.output_dir) 


    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        train_dataloader.sampler.set_epoch(epoch)


        train_stats = train_grad_acc(model, model_guideline, image_encoder, text_encoder, tokenizer, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args, config, writer, config['grad_accumulation_steps'], args.guideline_path) 

        for k, v in train_stats.items():
            if k == 'loss':
                train_loss_epoch = v
            elif k == 'loss_ce':
                train_loss_ce_epoch = v
            elif k == 'loss_clip':
                train_loss_clip_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/train_loss_ce_epoch', float(train_loss_ce_epoch), epoch)
        writer.add_scalar('loss/train_loss_clip_epoch', float(train_loss_clip_epoch), epoch)
        writer.add_scalar('lr/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)


        valid_on_ICD(model, model_guideline, image_encoder, text_encoder, tokenizer, val_dataloader,epoch,device,args,config, args.guideline_path)

        if utils.is_main_process() and (epoch+1)%10==0 :  
            save_obj = {
                    'model': model.state_dict(),
                    'model_guideline': model_guideline.state_dict(),
                    'image_encoder': image_encoder.state_dict(),
                    'text_encoder':text_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pt'))
            
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Evaluating')
    test_logits(args, config, max_epoch)



if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--dataset', default='icd53')
    parser.add_argument('--config', default='./configs/ICD.yaml')

    parser.add_argument('--class_num', default=1, type=int)
    # Port
    parser.add_argument('--port', default=80, type=int)

    parser.add_argument('--loss_ratio', default=1, type=float)
    parser.add_argument('--contrast_ratio_text', default=0.1, type=float)
    parser.add_argument('--temperature_text', default=0.5, type=float)
    parser.add_argument('--contrast_ratio_vision', default=0.1, type=float)
    parser.add_argument('--temperature_vision', default=2.0, type=float)

    parser.add_argument('--dist_backend', default='nccl')

    parser.add_argument('--output_dir', default='./output_dir/0116_toy')
    parser.add_argument('--image_encoder_name', default='resnet50')

    parser.add_argument('--guideline_path', default='')
    parser.add_argument('--bert_model_name', default='')
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--embed_dim', type=int, default=768, help='embedding dim')
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--gpu', default='0', type=str, help='gpu')
    args = parser.parse_args()
    os.environ['MASTER_PORT'] = f'{args.port}'

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  

    logging.info("Params:")
    params_file = os.path.join(args.output_dir, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    seed_torch(args.seed)
    main(args, config)
