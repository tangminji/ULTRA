import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Our_SupCL_Origin_loss(nn.Module):
    '''
    modified based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    '''
    def __init__(self, args, contrast_mode='all', base_temperature=0.07):
        super(Our_SupCL_loss, self).__init__()
        self.temperature = args.temp
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.c = args.c
        self.device = args.device
        self.gamma = args.gamma
        self.eta = args.eta
        self.warmup = args.warm_up

    def forward(self, features, logits, labels, index, delta_smooth, epoch):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        #TODO: l_classification
        delta_smooth = delta_smooth.to(self.device)
        logits_norm = F.softmax(logits, dim=1)
        if len(labels.size()) == 1:
            db_labels = labels.repeat(2)
            target_oh = torch.zeros(db_labels.size(0), self.c + 1).to(self.device).scatter_(1, db_labels.view(-1, 1), (
                        1 - delta_smooth[index].repeat(2)).view(-1, 1))  # convert label to one-hot
            target_oh[:, -1] = delta_smooth[index].repeat(2)
        # calculate as cross-entropy loss
        loss_cla = -torch.mean(torch.sum(torch.log(logits_norm) * target_oh, dim=1))

        #TODO: l_constrativeloss
        if epoch >= self.warmup:
            if len(features.shape) < 3:
                raise ValueError('`features` needs to be [bsz, n_views, ...],'
                                 'at least 3 dimensions are required')
            if len(features.shape) > 3:
                features = features.view(features.shape[0], features.shape[1], -1)

            batch_size = features.shape[0]
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)

            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            if self.contrast_mode == 'one':
                anchor_feature = features[:, 0]
                anchor_count = 1
            elif self.contrast_mode == 'all':
                anchor_feature = contrast_feature
                anchor_count = contrast_count
            else:
                raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # tile mask
            mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
                0
            )
            # TODO: newly added, mask-out high uncertainty cases
            low_unc = logits_norm[:, -1] <= self.eta
            low_uncertain_mask = torch.zeros_like(mask)
            low_uncertain_mask[low_unc] = True
            low_uncertain_mask.float().to(self.device)

            mask = mask * low_uncertain_mask
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            # for numerical stability
            tmp = 1e-8
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+tmp)

            # loss
            loss_cl = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss_cl = loss_cl.view(anchor_count, batch_size).mean()
        else:
            loss_cl = 0

        #TODO: combine loss_cla and loss_cl
        loss = loss_cla + self.gamma * loss_cl
        return loss

class Our_SupCL_loss(nn.Module):
    '''
    modified based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    '''
    def __init__(self, args, contrast_mode='all', base_temperature=0.07):
        super(Our_SupCL_loss, self).__init__()
        self.temperature = args.temp
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.c = args.c
        self.device = args.device
        # self.gamma = args.gamma
        gamma_schedule = torch.ones(args.n_epoch)
        # Linear
        # gamma_schedule = torch.linspace(0, 1, args.warm_up)
        # Exponent
        gamma_schedule[:args.warm_up] = torch.logspace(-args.warm_up, 0, args.warmup, np.e)
        self.gamma = gamma_schedule * args.gamma
        self.eta = args.eta
        self.warmup = args.warm_up

    def forward(self, features, logits, labels, index, delta_smooth, epoch):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        #TODO: l_classification
        delta_smooth = delta_smooth.to(self.device)
        logits_norm = F.softmax(logits, dim=1)
        if len(labels.size()) == 1:
            db_labels = labels.repeat(2)
            target_oh = torch.zeros(db_labels.size(0), self.c + 1).to(self.device).scatter_(1, db_labels.view(-1, 1), (
                        1 - delta_smooth[index].repeat(2)).view(-1, 1))  # convert label to one-hot
            target_oh[:, -1] = delta_smooth[index].repeat(2)
        # calculate as cross-entropy loss
        loss_cla = -torch.mean(torch.sum(torch.log(logits_norm) * target_oh, dim=1))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                                'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        # TODO: newly added, mask-out high uncertainty cases
        low_unc = logits_norm[:, -1] <= self.eta
        low_uncertain_mask = torch.zeros_like(mask)
        low_uncertain_mask[low_unc] = True
        low_uncertain_mask.float().to(self.device)

        mask = mask * low_uncertain_mask
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # for numerical stability
        tmp = 1e-8
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+tmp)

        # loss
        loss_cl = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_cl = loss_cl.view(anchor_count, batch_size).mean()

        #TODO: combine loss_cla and loss_cl
        loss = loss_cla + self.gamma[epoch] * loss_cl
        return loss

class CELoss(nn.Module):
    '''
    CE, class_num=10
    '''
    def __init__(self, num_class, device):
        super(CELoss, self).__init__()
        self.c = num_class
        self.device = device

    def forward(self, logits, targets):
        logits_norm = F.softmax(logits, dim=1)
        if len(targets.size())==1:
            target_oh = torch.zeros(targets.size(0), self.c).to(self.device).scatter_(1, targets.view(-1,1), 1) # convert label to one-hot

        loss = -torch.mean(torch.sum(torch.log(logits_norm) * target_oh, dim=1))
        return loss

class CE_OurLoss(nn.Module):
    '''
    CE for ours, class_num=11
    '''
    def __init__(self, num_class, device):
        super(CE_OurLoss, self).__init__()
        self.c = num_class
        self.device = device

    def forward(self, logits, targets):
        logits_norm = F.softmax(logits, dim=1)
        if len(targets.size())==1:
            target_oh = torch.zeros(targets.size(0), self.c).to(self.device).scatter_(1, targets.view(-1,1), 1) # convert label to one-hot

        loss = -torch.mean(torch.sum(torch.log(logits_norm[:,:-1]) * target_oh, dim=1))
        return loss

class OurLoss(nn.Module):
    '''
    our loss with regularization
    '''
    def __init__(self, num_class, device):
        super(OurLoss, self).__init__()
        self.c = num_class
        self.device = device
        #self.epsilon = sys.float_info.epsilon
        #self.init_lambda = init_lambda
        #self.lamb = lamb
        #self.max_eps = max_eps

    # def exponential_decay_lambda(self, cur_eps, max_eps, initial_lambda):
    #     lambda_val = initial_lambda * (0.1 ** (cur_eps / max_eps))
    #     lambda_val = max(lambda_val, 0)
    #     return lambda_val
    #
    # def linear_decay_lambda(self, cur_eps, max_eps, initial_lambda):
    #     #keep max_eps > 0
    #     if max_eps < self.epsilon:
    #         max_eps = self.epsilon
    #     lambda_val = max(initial_lambda - (cur_eps / max_eps) * initial_lambda, 0)
    #     return lambda_val

    def forward(self, logits, targets, index, delta_smooth):
        delta_smooth = delta_smooth.to(self.device)
        logits_norm = F.softmax(logits, dim=1)
        if len(targets.size()) == 1:
            target_oh = torch.zeros(targets.size(0), self.c+1).to(self.device).scatter_(1, targets.view(-1,1), (1-delta_smooth[index]).view(-1,1))  # convert label to one-hot
            target_oh[:,-1] = delta_smooth[index]
        #calculate as cross-entropy loss
        loss = -torch.mean(torch.sum(torch.log(logits_norm) * target_oh, dim=1))


        # ce+regularization term
        #l_ce = -torch.sum(torch.log(logits_norm[:, :-1]) * target_oh, dim=1)
        # args.lam should not be too large, keep l_reg smaller(or comparable to) l_ce
        #l_reg = torch.sum(self.lamb * (1 - logits_norm[:, :-1]), dim=1)
        # lamb = self.linear_decay_lambda(cur_eps, self.max_eps, self.init_lambda)
        # l_reg = torch.sum(lamb * (1 - logits_norm[:, :-1]), dim=1)
        # l_reg = torch.sum(
        #     lamb * (1 - logits_norm[:, :-1]) / logits_norm[:, -1].unsqueeze(1).expand_as(logits_norm[:, :-1]),
        #     dim=1)
        # if cur_eps < self.max_eps:
        #     loss = torch.mean(l_ce + l_reg)
        # else:
        #     loss = torch.mean(l_ce)
        return loss


