import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F
import math

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cal_organ_loss = nn.CrossEntropyLoss()
        #self.criterionCons = nn.KLDivLoss(reduction='batchmean')



    def forward(self, net_output: torch.Tensor, organ_output: torch.Tensor, discrim_tumor, discrim_latent, target: torch.Tensor):
        


        tumor_loss = self.cal_loss_single(net_output, target)
        organ_loss = self.cal_loss_single(organ_output, target)

        # adversial loss
        if discrim_tumor is None:
            diversity_loss = F.mse_loss(discrim_tumor, discrim_latent) 
   
            total_loss = tumor_loss + organ_loss + 0.1 * diversity_loss
        else:
            total_loss = tumor_loss + organ_loss

                

        return total_loss

    def cal_loss_single(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        #self._nb_current = target.shape[0]

        net_output = net_output[: target.shape[0]]
        #print('########### net_output shape: #############', net_output.shape)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
    
    def loss_consistency_mse(self, pred_all):
    
        lambda_consist = 0.1  
        
        batch_size = self._nb_current
        pred_all_prob = F.softmax(pred_all, dim=1)
        
        group1 = pred_all_prob[:batch_size]
        group2 = pred_all_prob[batch_size:batch_size*2]
        group3 = pred_all_prob[batch_size*2:]
        
        pred_avg = (group1 + group2 + group3) / 3.0
        
        pred_avg_expanded = torch.cat([pred_avg, pred_avg, pred_avg], dim=0)

        loss_consist = F.mse_loss(pred_all_prob, pred_avg_expanded.detach(), reduction='mean')
        
        return lambda_consist * loss_consist

    def kl_loss(self, pred_all, pred_avg):
        #self.criterionCons = nn.KLDivLoss(reduction='batchmean')
        loss_scores = F.kl_div(pred_all, pred_avg, reduction='batchmean')
        return loss_scores

    def loss_consistency(self, pred_all):
        '''
        KL-term, enforcing conditional distribution remains unchanged regardless of interventions applied
        '''
        
        #lambda_consist = 10
        lambda_consist = 0.1

        pred_all_prob = F.softmax(pred_all, dim = 1)
        pred_avg = 1.0 / 3 * ( pred_all_prob[: self._nb_current] + pred_all_prob[self._nb_current : self._nb_current * 2] + pred_all_prob[self._nb_current * 2: ]) # efficient implementation inspired by Xu et al. (Randconv)
        pred_avg = torch.cat([pred_avg  for ii in range(3)], dim = 0)
        pred_all = F.log_softmax(pred_all, dim = 1) # according to pytorch 1.3 documentation, input is log_prob, target is prob
        loss_consist = self.kl_loss(pred_all, pred_avg)
        

        self.loss_consist = lambda_consist * loss_consist
        #self.loss_consist_tr = self.loss_consist.data

        return self.loss_consist
    
    def cross_entropy_loss(self, logits, target):

        target = target.long()
        one_hot_target = F.one_hot(target, num_classes=2)
       
        # 使用 log_softmax + nll_loss 组合
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, one_hot_target.argmax(dim=1))
        
        return loss

    def adjust_loss(self, loss_organ, loss_tumor):
        L_th = -0.7
        lambda_ini = 0.99
        if loss_organ > L_th:
            lambda_t = lambda_ini
        else:
            lambda_t = 1 - lambda_ini

        loss_all = lambda_t * loss_organ + (1 - lambda_t) * loss_tumor
        print('Loss Organ: ', loss_organ, 'Loss Tumor: ', loss_tumor)
        print('Lambda Loss Organ: ', lambda_t, 'Lambda Loss tumor: ', (1-lambda_t))

        return loss_all
    def update_temperature(self):
        
        if self.step_count == 0:
            self.current_temp = self.temp_ini
        else:
            new_temp = self.current_temp * self.decay_rate
            if new_temp >= self.temp_fin:
                self.current_temp = new_temp
            # 否则保持当前温度不变
        
        self.step_count += 1
        
    def calculate_gamma(self, loss_tumor):
        
        if self.current_temp * self.decay_rate < self.temp_fin:
            # 训练后期，使用 gamma_str 防止剧烈变化
            gamma = min(math.exp(-1 / (loss_tumor.item() + self.epsilon) / self.current_temp), 
                       self.gamma_str)
        else:
            gamma = math.exp(-1 / (loss_tumor.item() + self.epsilon) / self.current_temp)
        
        return gamma
    
    def calculate_adaptive_weights(self, loss_organ, loss_tumor):
        
        gamma = self.calculate_gamma(loss_tumor)
        
        # 基于gamma动态调整权重
        # 当gamma小时，更关注organ loss; 当gamma大时，更关注tumor loss
        organ_weight = self.organ_base_weight * (1 - gamma)
        tumor_weight = self.tumor_base_weight * gamma
        
        return organ_weight, tumor_weight, gamma
    
    def calculate_tumor_weight(self):
        
        self.progress = min(1.0, self.current_epoch / self.max_epochs)
        
        if self.growth_type == 'sigmoid':
            
            tumor_factor = 1 / (1 + math.exp(-10 * (self.progress - 0.5)))
        elif self.growth_type == 'quadratic':
            
            tumor_factor = self.progress ** 2
        elif self.growth_type == 'exponential':
            # 指数增长，越往后增长越快
            tumor_factor = (math.exp(self.progress) - 1) / (torch.exp(torch.tensor(1.0)) - 1)
        else:
            # 默认线性
            tumor_factor = self.progress
        
        #tumor_weight = 1.0 + tumor_factor * (self.tumor_max_weight - 1.0)
        self.tumor_weight = tumor_factor
        organ_weight = 1 - self.tumor_weight

        #self.current_epoch = self.current_epoch + 1

        return self.tumor_weight, organ_weight
    def update_epoch_t(self):
        if self.tumor_weight > 0.95:
            self.current_epoch = 0
                
        self.current_epoch += 1



class SimulatedAnnealingOrganTumorLoss(nn.Module):
    def __init__(self, 
                 temp_ini=1.0,
                 temp_fin=0.01,
                 decay_rate=0.95,
                 gamma_str=0.1,
                 organ_base_weight=1.0,
                 tumor_base_weight=1.0,
                 epsilon=1e-8):
        """
        基于模拟退火的器官-肿瘤损失函数
        
        Args:
            temp_ini: 初始温度
            temp_fin: 最终温度
            decay_rate: 温度衰减率
            gamma_str: 防止肿瘤损失在训练后期急剧增加的约束参数
            organ_base_weight: 器官损失基础权重
            tumor_base_weight: 肿瘤损失基础权重
            epsilon: 数值稳定性参数
        """
        super().__init__()
        self.temp_ini = temp_ini
        self.temp_fin = temp_fin
        self.decay_rate = decay_rate
        self.gamma_str = gamma_str
        self.organ_base_weight = organ_base_weight
        self.tumor_base_weight = tumor_base_weight
        self.epsilon = epsilon
        
        # 初始化状态
        self.current_temp = temp_ini
        self.step_count = 0
        self.organ_loss_history = []
        self.tumor_loss_history = []
        
    def update_temperature(self):
        
        if self.step_count == 0:
            self.current_temp = self.temp_ini
        else:
            new_temp = self.current_temp * self.decay_rate
            if new_temp >= self.temp_fin:
                self.current_temp = new_temp
            # 否则保持当前温度不变
        
        self.step_count += 1
        
    def calculate_gamma(self, loss_tumor):
        
        if self.current_temp * self.decay_rate < self.temp_fin:
            # 训练后期，使用 gamma_str 防止剧烈变化
            gamma = min(math.exp(-1 / (loss_tumor.item() + self.epsilon) / self.current_temp), 
                       self.gamma_str)
        else:
            gamma = math.exp(-1 / (loss_tumor.item() + self.epsilon) / self.current_temp)
        
        return gamma
    
    def calculate_adaptive_weights(self, loss_organ, loss_tumor):
        
        gamma = self.calculate_gamma(loss_tumor)
        
        # 基于gamma动态调整权重
        # 当gamma小时，更关注organ loss; 当gamma大时，更关注tumor loss
        organ_weight = self.organ_base_weight * (1 - gamma)
        tumor_weight = self.tumor_base_weight * gamma
        
        return organ_weight, tumor_weight, gamma
    
    def forward(self, loss_organ, loss_tumor):
        
        
        self.organ_loss_history.append(loss_organ.item())
        self.tumor_loss_history.append(loss_tumor.item())
        if len(self.organ_loss_history) > 50:
            self.organ_loss_history.pop(0)
        if len(self.tumor_loss_history) > 50:
            self.tumor_loss_history.pop(0)
        
        # 计算自适应权重
        organ_weight, tumor_weight, gamma = self.calculate_adaptive_weights(loss_organ, loss_tumor)
        
        # 计算总损失
        total_loss = organ_weight * loss_organ + tumor_weight * loss_tumor
        
        # 更新温度
        self.update_temperature()
        
        return total_loss


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
