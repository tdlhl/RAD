import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        
    def forward(self, image_features, text_features):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        labels = torch.eye(num_logits, device=device, dtype=torch.float)
        pred_1 = F.log_softmax(logits_per_image,dim=-1)
        pred_2 = F.log_softmax(logits_per_text,dim=-1)
        loss_a = F.kl_div(pred_1, labels,reduction = 'sum')/num_logits
        loss_b = F.kl_div(pred_2, labels,reduction = 'sum')/num_logits
        total_loss = (loss_a + loss_b)/2
        return total_loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1, neg_ratio=5):
        super().__init__()
        self.temperature = temperature
        self.neg_ratio = neg_ratio

    def forward(self, features, prototypes, labels):
        """
        Args:
            features: (bs, dim)
            prototypes: (num_classes, dim)
            labels: (bs, num_classes)
        """
        device = features.device
        bs, num_classes = labels.shape
        features = F.normalize(features, p=2, dim=-1)  # (bs, dim)
        prototypes = F.normalize(prototypes, p=2, dim=-1)  # (num_classes, dim)
        
        sim_matrix = torch.mm(features, prototypes.transpose(0, 1)) / self.temperature
        
        losses = []
        for i in range(bs):
            pos_mask = labels[i].bool()  # (num_classes,)
            neg_mask = ~pos_mask
            
            if pos_mask.sum() == 0:
                continue

            pos_sim = sim_matrix[i][pos_mask]
            
            num_neg = min(self.neg_ratio * pos_mask.sum(), neg_mask.sum())
            neg_indices = torch.multinomial(neg_mask.float(), num_neg)
            neg_sim = sim_matrix[i][neg_indices]
            
            logits = torch.cat([pos_sim, neg_sim])
            targets = torch.zeros_like(logits)
            targets[:len(pos_sim)] = 1.0 / len(pos_sim)
            loss = F.binary_cross_entropy_with_logits(
                logits, 
                targets,
                reduction='mean'
            )
            losses.append(loss)
        
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=device)


class SupConLossPositiveOnly(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, labels):
        """
        Args:
            features: (bs, dim)
            prototypes: (num_classes, dim)
            labels: (bs, num_classes)
        """
        device = features.device
        bs, num_classes = labels.shape

        features = F.normalize(features, p=2, dim=-1)  # (bs, dim)
        prototypes = F.normalize(prototypes, p=2, dim=-1)  # (num_classes, dim)
        
        sim_matrix = torch.mm(features, prototypes.transpose(0, 1)) / self.temperature
        
        losses = []
        for i in range(bs):
            pos_mask = labels[i].bool()  # (num_classes,)
            
            if pos_mask.sum() == 0:
                continue
            
            pos_sim = sim_matrix[i][pos_mask]
            
            targets = torch.ones_like(pos_sim)
            loss = F.binary_cross_entropy_with_logits(
                pos_sim, 
                targets,
                reduction='mean'
            )
            losses.append(loss)
        
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=device)