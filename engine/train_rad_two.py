import json
import logging
import math
import os
import cv2
import re
import time
import numpy as np
from torch.distributed import ReduceOp
import random

from PIL import Image
from contextlib import suppress
from itertools import chain
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, precision_recall_curve,recall_score
import contextlib

import torch
import torch.nn.functional as F
from torch import nn

from factory import utils
from factory.loss import ClipLoss, SupConLoss


def Shuffle_Batch_Data(data_in):
    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]

def Combine_AmplitudeANDPhase(amp, phe):
    return torch.mul(amp, torch.exp(1j*phe))

def mixup_data(x, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    y = Shuffle_Batch_Data(x)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x

def FFT2_Amp_MixUp(data_original, data_aug, lamda):
    fft_data_original = torch.fft.fft2(data_original)
    fft_data_aug = torch.fft.fft2(data_aug)
    
    aug_amp = lamda*torch.abs(fft_data_original) + (1-lamda)*torch.abs(fft_data_aug)
    fft_mixup_data = torch.mul(aug_amp, torch.exp(1j*torch.angle(fft_data_original)))
    return torch.real(torch.fft.ifft2(fft_mixup_data))

def fourier_aug(batch_data, p=0.5):
    batch_x = batch_data
    batch_y = Shuffle_Batch_Data(batch_data)
    apply_p = np.random.rand()
    if apply_p<=p:
        lamda_vector = np.random.rand(batch_x.size(0))
        for i in range(batch_x.size(0)):
            batch_x[i] = FFT2_Amp_MixUp(batch_x[i], batch_y[i], lamda_vector[i])
        return batch_x
    else:
        return batch_x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_text_features_bert(model,text_list,tokenizer,device,max_length):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    text_token = tokenizer(
        list(text_list),
        add_special_tokens=True,  
        padding='max_length',   
        truncation=True,     
        max_length=max_length,  
        return_tensors="pt"  
    ).to(device=device)

    text_features, text_last_hidden_state = model.encode_text(text_token)

    return text_features, text_last_hidden_state

def train_grad_acc(model, model_guideline, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer, accumulation_steps, guideline_path):
    clip_loss = ClipLoss()
    contrast_loss_text = SupConLoss(temperature=args.temperature_text, neg_ratio=5)
    contrast_loss_vision = SupConLoss(temperature=args.temperature_vision, neg_ratio=5)
    
    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_contrast_text_m = AverageMeter()
    loss_contrast_vision_m = AverageMeter()
    loss_ce_m = AverageMeter()
    loss_ce_image_m = AverageMeter()
    loss_ce_guideline_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    model.train()  
    model_guideline.train() 
    image_encoder.train()  
    text_encoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce_image', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce_guideline', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_clip', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_contrast_text', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_contrast_vision', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 5
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)
    num_batches_per_epoch = data_loader.num_batches
    sample_digits = math.ceil(math.log(data_loader.num_samples + 1, 10))

    label_sequence_length = 16
    report_sequence_length = 512
    ehr_sequence_length = 512
    if 'Bio_ClinicalBERT' in args.bert_model_name:
        print('Using guideline_sequence_length 32!')
        guideline_sequence_length = 32
    else:
        guideline_sequence_length = 64

    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if i==4:
        #     raise RuntimeError('break when 5 iteration!')
        image = fourier_aug(sample['image'].to(device))
        label = sample['label'].long().to(device)

        entity = sample['entity']

        report_text = []
        ehr_text = []

        for text in entity:
            match = re.match(r'<Report>:\s*(.*?)\s*<EHR>:\s*(.*)', text, re.DOTALL)
            if match:
                report_text.append(match.group(1).strip())
                ehr_text.append(match.group(2).strip())
            else:
                raise RuntimeError('no match for <Report> and <EHR>!')

        data_time_m.update(time.time() - end)

        label_list = [
                "abscess of lung and mediastinum, pyothorax, mediastinitis",
                "acute bronchiolitis and unspecified acute lower respiratory infection",
                "bronchiectasis",
                "bronchitis",
                "chronic airway obstruction, not elsewhere classified",
                "complications and ill-defined descriptions of heart disease, cardiomegaly",
                "extrinsic asthma, unspecified",
                "farmers' lung and pneumoconiosis due to asbestos and mineral fibers",
                "flu due to avian flu virus",
                "fracture of rib(s), sternum and thoracic spine",
                "lung opacity",
                "other pleural conditions",
                "other respiratory disorders, consolidation",
                "pleurisy without mention of effusion or current tuberculosis",
                "pneumoconiosis",
                "pneumonia",
                "pneumonitis due to solids, liquids, and radiation",
                "pulmonary collapse",
                "pulmonary congestion, edema, and pleural effusion",
                "pulmonary emphysema and emphysematous bleb",
                "pulmonary fibrosis and other interstitial pulmonary diseases",
                "surgical instruments, materials and cardiovascular devices",
                "traumatic, spontaneous, and tension pneumothorax, air leak",
                "hypertension",
                "hyperosmolality and/or hypernatremia",
                "diabetes mellitus",
                "heart failure",
                "tachycardia",
                "chronic total occlusion of coronary artery",
                "hypotension",
                "myocardial infarction",
                "chronic ischemic heart disease",
                "atrial fibrillation and flutter",
                "primary pulmonary hypertension",
                "acute posthemorrhagic anemia",
                "endomyocardial fibrosis",
                "other coagulation defects",
                "malignant neoplasm of trachea",
                "atherosclerosis of aorta",
                "abnormal serum enzyme levels",
                "cardiomyopathy",
                "other cardiac arrhythmias",
                "vitamin D deficiency",
                "esophageal varices",
                "angina decubitus",
                "pulmonary embolism",
                "poisoning by, adverse effect of and underdosing of agents primarily affecting the cardiovascular system",
                "acute and subacute bacterial endocarditis",
                "polyarteritis nodosa",
                "arterial embolism and thrombosis",
                "acute and subacute endocarditis",
                "phlebitis and thrombophlebitis",
                "thalassemia",
            ]

        if guideline_path:
            guideline_list = []
            guideline_vision_parts = []
            with open(guideline_path, 'r') as guideline_jsonl:
                for idx, line in enumerate(guideline_jsonl):
                    current_line = json.loads(line)
                    if current_line['query']==label_list[idx]:
                        title_content = current_line['guideline_1_content']
                        split_parts = title_content.split('\n\n')
                        if len(split_parts) > 3 and 'Radiological' in split_parts[3]:
                            vision_part = split_parts[3]
                        else:
                            find_vision = False
                            for current_part in split_parts:
                                if 'Radiological' in current_part:
                                    vision_part = current_part
                                    find_vision = True
                                    break
                            if not find_vision:
                                for current_part in split_parts:
                                    if 'Imaging' in current_part:
                                        vision_part = current_part
                                        find_vision = True
                                        break
                                if not find_vision:
                                    vision_part = split_parts[-1] 
                        guideline_list.append(title_content)
                        guideline_vision_parts.append(vision_part)
                    else:
                        raise RuntimeError('guideline do not match label!')
        if args.bert_model_name:
            label_features_pool, label_last_hidden_state  = get_text_features_bert(text_encoder,label_list,tokenizer,device,max_length=args.max_length)
            report_features_pool, report_last_hidden_state  = get_text_features_bert(text_encoder,report_text,tokenizer,device,max_length=args.max_length)
            ehr_features_pool, ehr_last_hidden_state  = get_text_features_bert(text_encoder,ehr_text,tokenizer,device,max_length=args.max_length)
            if guideline_path:
                guideline_features_pool, guideline_last_hidden_state  = get_text_features_bert(text_encoder,guideline_list,tokenizer,device,max_length=args.max_length)
                with torch.no_grad():
                    guideline_vision_features_pool, _  = get_text_features_bert(text_encoder,guideline_vision_parts,tokenizer,device,max_length=args.max_length)

        image_features, image_features_pool = image_encoder(image)

        label_features = label_last_hidden_state[:, :label_sequence_length, :]
        report_features = report_last_hidden_state[:, :report_sequence_length, :]
        ehr_features = ehr_last_hidden_state[:, :ehr_sequence_length, :]

        fusion_features = torch.cat((image_features, report_features, ehr_features), dim=1)
        pred_class_image = model(label_features, fusion_features)

        loss_contrast_text = contrast_loss_text(
            features=report_features_pool,  # (bs,1,768)
            prototypes=guideline_features_pool,
            labels=label.float()  # (bs,53)
        )
        loss_contrast_vision = contrast_loss_vision(
            features=image_features_pool,  # (bs,1,768)
            prototypes=guideline_vision_features_pool,
            labels=label.float()  # (bs,53)
        )

        label = label.float()
        loss_ce_image = F.binary_cross_entropy_with_logits(pred_class_image.view(-1,1), label.view(-1,1))
        if guideline_path:
            guideline_features = guideline_last_hidden_state[:, :guideline_sequence_length, :]
            pred_class_guideline = model_guideline(guideline_features, fusion_features)
            loss_ce_guideline = F.binary_cross_entropy_with_logits(pred_class_guideline.view(-1,1), label.view(-1,1))
            loss_ce = loss_ce_image + loss_ce_guideline
        else:
            loss_ce = loss_ce_image
        loss_clip = clip_loss(image_features_pool, report_features_pool)

        loss = loss_ce + loss_clip * args.loss_ratio + loss_contrast_text * args.contrast_ratio_text + loss_contrast_vision * args.contrast_ratio_vision

        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:

            optimizer.step()
            optimizer.zero_grad()

            if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
                scheduler.step(i // step_size)

        writer.add_scalar('loss/loss', loss*accumulation_steps, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        if guideline_path:
            writer.add_scalar('loss/loss_ce_guideline', loss_ce_guideline, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        writer.add_scalar('loss/loss_contrast_text', loss_contrast_text, scalar_step)
        writer.add_scalar('loss/loss_contrast_vision', loss_contrast_vision, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item()*accumulation_steps)
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        if guideline_path:
            metric_logger.update(loss_ce_guideline=loss_ce_guideline.item())
        metric_logger.update(loss_clip=loss_clip.item())
        metric_logger.update(loss_contrast_text=loss_contrast_text.item())
        metric_logger.update(loss_contrast_vision=loss_contrast_vision.item())
       
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if i % 100 == 0:
            batch_size = len(image)
            num_samples = batch_count * batch_size
            samples_per_epoch = data_loader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            loss_m.update(loss.item()*accumulation_steps, batch_size)
            loss_clip_m.update(loss_clip.item(), batch_size)
            loss_contrast_text_m.update(loss_contrast_text.item(), batch_size)
            loss_contrast_vision_m.update(loss_contrast_vision.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if guideline_path:
                loss_ce_guideline_m.update(loss_ce_guideline.item(), batch_size)

                logging.info(
                        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                        f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                        f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                        f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                        f"Loss_ce_guideline: {loss_ce_guideline_m.val:#.5g} ({loss_ce_guideline_m.avg:#.4g}) "
                        f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                        f"loss_contrast_text: {loss_contrast_text_m.val:#.5g} ({loss_contrast_text_m.avg:#.4g}) "
                        f"loss_contrast_vision: {loss_contrast_vision_m.val:#.5g} ({loss_contrast_vision_m.avg:#.4g}) "
                        f"Data (t): {data_time_m.avg:.3f} "
                        f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                        f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                    )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                    f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def valid_on_ICD(model, model_guideline, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, guideline_path):
    model.eval()
    model_guideline.eval()
    image_encoder.eval()
    text_encoder.eval()

    text_list = [
                "abscess of lung and mediastinum, pyothorax, mediastinitis",
                "acute bronchiolitis and unspecified acute lower respiratory infection",
                "bronchiectasis",
                "bronchitis",
                "chronic airway obstruction, not elsewhere classified",
                "complications and ill-defined descriptions of heart disease, cardiomegaly",
                "extrinsic asthma, unspecified",
                "farmers' lung and pneumoconiosis due to asbestos and mineral fibers",
                "flu due to avian flu virus",
                "fracture of rib(s), sternum and thoracic spine",
                "lung opacity",
                "other pleural conditions",
                "other respiratory disorders, consolidation",
                "pleurisy without mention of effusion or current tuberculosis",
                "pneumoconiosis",
                "pneumonia",
                "pneumonitis due to solids, liquids, and radiation",
                "pulmonary collapse",
                "pulmonary congestion, edema, and pleural effusion",
                "pulmonary emphysema and emphysematous bleb",
                "pulmonary fibrosis and other interstitial pulmonary diseases",
                "surgical instruments, materials and cardiovascular devices",
                "traumatic, spontaneous, and tension pneumothorax, air leak",
                "hypertension",
                "hyperosmolality and/or hypernatremia",
                "diabetes mellitus",
                "heart failure",
                "tachycardia",
                "chronic total occlusion of coronary artery",
                "hypotension",
                "myocardial infarction",
                "chronic ischemic heart disease",
                "atrial fibrillation and flutter",
                "primary pulmonary hypertension",
                "acute posthemorrhagic anemia",
                "endomyocardial fibrosis",
                "other coagulation defects",
                "malignant neoplasm of trachea",
                "atherosclerosis of aorta",
                "abnormal serum enzyme levels",
                "cardiomyopathy",
                "other cardiac arrhythmias",
                "vitamin D deficiency",
                "esophageal varices",
                "angina decubitus",
                "pulmonary embolism",
                "poisoning by, adverse effect of and underdosing of agents primarily affecting the cardiovascular system",
                "acute and subacute bacterial endocarditis",
                "polyarteritis nodosa",
                "arterial embolism and thrombosis",
                "acute and subacute endocarditis",
                "phlebitis and thrombophlebitis",
                "thalassemia",
    ]

    if guideline_path:
            guideline_list = []
            guideline_vision_parts = []
            with open(guideline_path, 'r') as guideline_jsonl:
                for idx, line in enumerate(guideline_jsonl):
                    current_line = json.loads(line)
                    if current_line['query']==text_list[idx]:
                        title_content = current_line['guideline_1_content']
                        split_parts = title_content.split('\n\n')
                        if len(split_parts) > 3 and 'Radiological' in split_parts[3]:
                            vision_part = split_parts[3]
                        else:
                            find_vision = False
                            for current_part in split_parts:
                                if 'Radiological' in current_part:
                                    vision_part = current_part
                                    find_vision = True
                                    break
                            if not find_vision:
                                for current_part in split_parts:
                                    if 'Imaging' in current_part:
                                        vision_part = current_part
                                        find_vision = True
                                        break
                                if not find_vision:
                                    vision_part = split_parts[-1] 
                        guideline_list.append(title_content)
                        guideline_vision_parts.append(vision_part)
                    else:
                        raise RuntimeError('guideline do not match label!')

    if args.bert_model_name:
        label_features_pool, label_last_hidden_state  = get_text_features_bert(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
        if guideline_path:
            guideline_features_pool, guideline_last_hidden_state  = get_text_features_bert(text_encoder,guideline_list,tokenizer,device,max_length=args.max_length)

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    pred_guideline = torch.FloatTensor().cuda()
    pred_avg = torch.FloatTensor().cuda()

    label_sequence_length = 16
    report_sequence_length = 512
    ehr_sequence_length = 512
    if 'Bio_ClinicalBERT' in args.bert_model_name:
        guideline_sequence_length = 32
    else:
        guideline_sequence_length = 64

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        entity = sample['entity']

        report_text = []
        ehr_text = []

        for text in entity:
            match = re.match(r'<Report>:\s*(.*?)\s*<EHR>:\s*(.*)', text, re.DOTALL)
            if match:
                report_text.append(match.group(1).strip())
                ehr_text.append(match.group(2).strip())
            else:
                raise RuntimeError('no match for <Report> and <EHR>!')

        label = label.float()
        gt = torch.cat((gt, label), 0)
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            report_features_pool, report_last_hidden_state  = get_text_features_bert(text_encoder,report_text,tokenizer,device,max_length=args.max_length)
            ehr_features_pool, ehr_last_hidden_state  = get_text_features_bert(text_encoder,ehr_text,tokenizer,device,max_length=args.max_length)

            label_features = label_last_hidden_state[:, :label_sequence_length, :]
            report_features = report_last_hidden_state[:, :report_sequence_length, :]
            ehr_features = ehr_last_hidden_state[:, :ehr_sequence_length, :]

            fusion_features = torch.cat((image_features, report_features, ehr_features), dim=1)

            pred_class = model(label_features, fusion_features)
            if guideline_path:
                guideline_features = guideline_last_hidden_state[:, :guideline_sequence_length, :]
                pred_class_guideline = model_guideline(guideline_features, fusion_features)
                label_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
                guideline_loss = F.binary_cross_entropy_with_logits(pred_class_guideline.view(-1,1),label.view(-1, 1))
                val_loss = label_loss + guideline_loss
                pred_class_guideline = torch.sigmoid(pred_class_guideline)
                pred_guideline = torch.cat((pred_guideline, pred_class_guideline[:,:,0]), 0)

                pred_class_avg = (pred_class_guideline+pred_class)/2
                pred_class_avg = torch.sigmoid(pred_class_avg)
                pred_avg = torch.cat((pred_avg, pred_class_avg[:,:,0]), 0)
            else:
                val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
            pred_class = torch.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class[:,:,0]), 0)

    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()

    pred_file_name = f"pred_epoch_{epoch}.npy" 
    pred_file_path = os.path.join(args.output_dir, pred_file_name)
    np.save(pred_file_path, pred_np)

    gt_file_name = f"gt_epoch_{epoch}.npy" 
    gt_file_path = os.path.join(args.output_dir, gt_file_name)
    np.save(gt_file_path, gt_np)

    if guideline_path:
        pred_guideline_np = pred_guideline.cpu().numpy()
        pred_guideline_file_name = f"pred_guideline_epoch_{epoch}.npy" 
        pred_guideline_file_path = os.path.join(args.output_dir, pred_guideline_file_name)
        np.save(pred_guideline_file_path, pred_guideline_np)

        pred_avg_np = pred_avg.cpu().numpy()
        pred_avg_file_name = f"pred_avg_epoch_{epoch}.npy" 
        pred_avg_file_path = os.path.join(args.output_dir, pred_avg_file_name)
        np.save(pred_avg_file_path, pred_avg_np)


def test_logits(args, config, max_epoch):
    if 'fair_ori' in args.dataset:
        n_class = 1
    elif 'skin' in args.dataset:
        n_class = 50
    elif 'nacc' in args.dataset:
        n_class = 11
    else:
        n_class = 53
    best_epoch = 0
    best_metrics = 0
    for i in range(0, max_epoch):
        gt_file_name = f"gt_epoch_{i}.npy" 
        gt_file_path = os.path.join(args.output_dir, gt_file_name)
        pred1_file_name = f"pred_epoch_{i}.npy" 
        pred1_file_path = os.path.join(args.output_dir, pred1_file_name)
        pred2_file_name = f"pred_guideline_epoch_{i}.npy" 
        pred2_file_path = os.path.join(args.output_dir, pred2_file_name)

        gt = np.load(gt_file_path)
        logits1 = np.load(pred1_file_path)
        logits2 = np.load(pred2_file_path)

        current_metrics_1 = compute_metrics(gt, logits1, n_class)
        current_metrics_2 = compute_metrics(gt, logits2, n_class)
        avg_metrics_1 = np.mean([np.mean(current_metrics_1[0]), current_metrics_1[2], np.mean(current_metrics_1[3]), 
                                np.mean(current_metrics_1[4]), np.mean(current_metrics_1[5]), np.mean(current_metrics_1[6]), current_metrics_1[7]])
        avg_metrics_2 = np.mean([np.mean(current_metrics_2[0]), current_metrics_2[2], np.mean(current_metrics_2[3]), 
                                np.mean(current_metrics_2[4]), np.mean(current_metrics_2[5]), np.mean(current_metrics_2[6]), current_metrics_2[7]])
        high = max(avg_metrics_1, avg_metrics_2)
        if high>best_metrics:
            best_epoch = i
            best_metrics = high
            best_gt = gt
            best_logits1 = logits1
            best_logits2 = logits2

    evaluate_combined_logits(best_gt, best_logits1, best_logits2, n_class, args)


def compute_AUCs(gt, pred, n_class):
    AUROCs = []
    for i in range(n_class):
        try:
            AUROCs.append(roc_auc_score(gt[:, i], pred[:, i]))
        except ValueError:
            AUROCs.append(0)
    return AUROCs

# Function to compute Average Precisions
def compute_APs(gt, pred, n_class):
    APs = []
    for i in range(n_class):
        try:
            APs.append(average_precision_score(gt[:, i], pred[:, i]))
        except ValueError:
            APs.append(0)
    mean_AP = np.mean(APs)
    return APs, mean_AP

# Function to compute metrics
def compute_metrics(gt, pred, n_class):
    pred_logits = pred
    aucs = compute_AUCs(gt, pred_logits, n_class)
    aps, maps = compute_APs(gt, pred_logits, n_class)  # aps: list of APs, maps: mean AP

    acc = []
    threshold = []
    precision_list = []
    recall_list = []
    max_f1s = []
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []

    for i in range(n_class):
        gt_np = gt[:, i]
        pred_np = pred_logits[:, i]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        f1_scores = (2 * precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        max_f1 = np.max(f1_scores)
        max_f1s.append(max_f1)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        threshold.append(max_f1_thresh)
        
        precision_list.append(precision[np.argmax(f1_scores)])
        recall_list.append(recall[np.argmax(f1_scores)])
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(gt_np, pred_np >= max_f1_thresh).ravel()
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
        
        # Compute accuracy for each class
        acc.append(accuracy_score(gt_np, pred_np >= max_f1_thresh))

    # Compute Example-based Accuracy
    pred_binary = (pred_logits >= np.array(threshold).reshape(1, -1)).astype(int)

    # Compute Subset Accuracy
    new_pred_binary = (pred_logits >= np.array(threshold)).astype(int)
    subset_acc = compute_subset_accuracy(gt.astype(int), new_pred_binary)

    return aucs, aps, maps, acc, max_f1s, precision_list, recall_list, subset_acc, threshold

def compute_subset_accuracy(gt, pred):
    return np.mean(np.all(gt == pred, axis=1))

def evaluate_combined_logits(gt, logits1, logits2, n_class, args):
    weights = np.arange(0, 1.02, 0.1)
    best_weights = np.ones(n_class)
    
    for class_idx in range(n_class):
        best_avg_metric = -1
        best_weight_for_class = 1
        
        for w in weights:
            current_weights = best_weights.copy()
            current_weights[class_idx] = w
            
            combined_logits = logits1 * current_weights.reshape(1, -1) + logits2 * (1 - current_weights).reshape(1, -1)
            
            metrics = compute_metrics(gt, combined_logits, n_class)
            avg_metrics = np.mean([np.mean(metrics[0]), metrics[2], np.mean(metrics[3]), 
                                   np.mean(metrics[4]), np.mean(metrics[5]), np.mean(metrics[6]), metrics[7]])

            if avg_metrics > best_avg_metric:
                best_avg_metric = avg_metrics
                best_weight_for_class = w
        
        best_weights[class_idx] = best_weight_for_class
    
    final_combined_logits = logits1 * best_weights.reshape(1, -1) + logits2 * (1 - best_weights).reshape(1, -1)
    final_metrics = compute_metrics(gt, final_combined_logits, n_class)
    final_path = os.path.join(args.output_dir, f"pred_fused.npy")

    np.save(final_path, final_combined_logits)
    
    print("\nFinal Metrics:")
    print(f"AUCs: {np.mean(final_metrics[0]) * 100}")
    print(f"Mean AP: {final_metrics[2] * 100}")
    print(f"Accuracies: {np.mean(final_metrics[3]) * 100}")
    print(f"Max F1 Scores: {np.mean(final_metrics[4]) * 100}")
    print(f"Precisions: {np.mean(final_metrics[5]) * 100}")
    print(f"Recalls: {np.mean(final_metrics[6]) * 100}")
    print(f"Subset Accuracy: {final_metrics[7] * 100}")