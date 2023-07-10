import torch
import torch.nn.functional as F
import numpy as np
from tensorboard_logger import configure, log_value#, log_histogram
#import tensorboard_logger
import random
from tqdm import tqdm
import os
import shutil
import time
import pywt
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import cv2

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def compute_topk_accuracy(prediction, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        prediction (torch.Tensor): N*C tensor, contains logits for N samples over C classes.
        target (torch.Tensor):  labels for each row in prediction.
        topk (tuple of int): different values of k for which top-k accuracy should be computed.

    Returns:
        result (tuple of float): accuracy at different top-k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = prediction.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result

def log(path, str):
    print(str)
    with open(path, 'a') as file:
        file.write(str)

def generate_log_dir(args):
    """Generate directory to save artifacts and tensorboard log files."""

    print('\nLog is going to be saved in: {}'.format(args.log_dir))

    if os.path.exists(args.log_dir):
        if args.restart:
            print('Deleting old log found in: {}'.format(args.log_dir))
            shutil.rmtree(args.log_dir)
            configure(args.log_dir, flush_secs=10)
        else:
            error='Old log found; pass --restart flag to erase'.format(args.log_dir)
            raise Exception(error)
    else:
        configure(args.log_dir, flush_secs=10)

def set_seed(args):
    """Set seed to ensure deterministic runs.

    Note: Setting torch to be deterministic can lead to slow down in training.
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def checkpoint(acc, epoch, net, save_dir, last=False):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not last:
        file_path = save_dir + '/net.pth'
    else:
        file_path = save_dir + '/net_last_ep.pth'
    torch.save(obj=state, f=file_path)

""" Training/testing """
# training
def train_ours(args, model, loader, optimizer, epoch, scheduler, criterion, net_record, delta_smooth):
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')#for classification
    t0 = time.time()

    for data, target, index in tqdm(loader, unit='batch'):
        if args.model_type == 'ours_cl':
            data = torch.cat([data[0], data[1]], dim=0)

        data, target = data.to(args.device), target.to(args.device)
        output, fea = model(data, filter=args.filter)

        if args.model_type == 'ours':
            loss = criterion(output, target, index, delta_smooth)
            # for correction
            net_record[epoch % args.rollWindow, index] = F.softmax(output.detach().cpu(), dim=1)

        elif args.model_type == 'ours_cl':
            bsz = target.shape[0]
            f1, f2 = torch.split(fea, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, output, target, index, delta_smooth, epoch)
            # for correction
            out1, out2 = torch.split(output, [bsz, bsz], dim=0)
            net_record[epoch % args.rollWindow, index] = F.softmax(out1.detach().cpu(), dim=1)
            net_record[args.rollWindow + epoch % args.rollWindow, index] = F.softmax(out2.detach().cpu(), dim=1)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), data.size(0))
        if args.model_type == 'ours_cl':
            target = target.repeat(2)
        acc1 = compute_topk_accuracy(output[:,:-1], target, topk=(1,))
        correct.update(acc1[0].item(), data.size(0))

    scheduler.step()
    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(args.logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.format(epoch, args.n_epoch, time.time() - t0, correct.avg, train_loss.avg))
    log_value('train/accuracy', correct.avg, step=epoch)
    return train_loss.avg, correct.avg


def lrt_correction_sr(args, y_gt, prediction):
    '''
    For label correction in serial mode, can be used for fine-tuning hyperparameters
    :param args
    :param y_gt: target labels
    :param prediction: prediction results
    '''
    y_cor = torch.tensor(y_gt).clone().to(args.device)
    y_hat, ind = prediction[:,:-1].max(dim=1)

    for i in range(len(y_cor)):
        #case 1: low uncertainty, high confidence, ID noise
        if ind[i] != y_cor[i] and float(y_hat[i]/prediction[i,-1]) > args.epsilon and prediction[i,-1] <= args.eta:
            y_cor[i] = ind[i]
        #case 2: high uncertainty, OOD noise
        elif prediction[i,-1] > args.eta:
            #pick a random label
            y_cor[i] = random.choice(range(args.c))
        #case 3: others, keep unchanged
        else:
            pass
    return y_cor

def lrt_correction_pr(args, epoch, y_tilde, prediction, delta_smooth):
    '''
    For label correction in parallel mode
    delta for each instance is updated seperately
    general parameter args.epsilon is updated
    :param args
    :param y_tilde: target labels
    :param prediction: prediction results (normalization finished)
    '''
    y_cor = torch.tensor(y_tilde).clone()
    y_hat, ind = prediction[:,:-1].max(dim=1)
    #case 1: no correction, only update delta_smooth
    delta_mask = ind == y_cor
    delta_smooth[delta_mask] -= args.inc
    delta_smooth[delta_mask] = torch.clamp(delta_smooth[delta_mask], min=0)

    # case 2: low uncertainty, high confidence, correct ID noise
    low_uncertainty_mask = (ind != y_cor) & (prediction[:, -1] <= args.eta) & (prediction[range(len(y_cor)),y_cor] / y_hat < args.epsilon)
    y_cor[low_uncertainty_mask] = ind[low_uncertainty_mask]

    corrected_ID_count = sum(low_uncertainty_mask)
    delta_smooth[low_uncertainty_mask] -= args.inc
    delta_smooth[low_uncertainty_mask] = torch.clamp(delta_smooth[low_uncertainty_mask], min=0)

    # case 3: high uncertainty, correct OOD noise
    high_uncertainty_mask = (ind != y_cor) & (prediction[:, -1] > args.eta)
    y_cor[high_uncertainty_mask] = torch.randint(args.c, (high_uncertainty_mask.sum(),))

    corrected_OOD_count = sum(high_uncertainty_mask)
    delta_smooth[high_uncertainty_mask] += args.inc
    delta_smooth[high_uncertainty_mask] = torch.clamp(delta_smooth[high_uncertainty_mask], max=0.5)

    print('Correct {} ID noise, {} OOD noise at {}-th epoch'.format(corrected_ID_count, corrected_OOD_count, epoch))
    if corrected_ID_count < 0.001*len(y_tilde):
        args.epsilon += 0.1#args.inc
        args.epsilon = min(args.epsilon, 0.9)

    return y_cor, delta_smooth

# training
def train_ce(args, model, loader, optimizer, epoch, scheduler, criterion):
    '''
    added by wtt
    :param args:
    :param model:
    :param loader:
    :param optimizer:
    :param epoch:
    :return:
    '''
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')  # for classification
    t0 = time.time()

    for data, target, index in tqdm(loader, unit='batch'):

        data, target = data.to(args.device), target.to(args.device)
        output = model(data, filter=args.filter)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), data.size(0))
        if len(target.size()) == 2:  # soft target
            target = target.argmax(dim=1, keepdim=True)
        acc1 = compute_topk_accuracy(output, target, topk=(1,))
        correct.update(acc1[0].item(), data.size(0))

    scheduler.step()
    log_value('train/lr', optimizer.param_groups[0]['lr'], step=epoch)
    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(args.logpath,
        'Time for Train-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.format(epoch, args.n_epoch, time.time() - t0, correct.avg,
                                                                      train_loss.avg))
    log_value('train/accuracy', correct.avg, step=epoch)
    return train_loss.avg, correct.avg

# testing
def evaluate(args, model, loader, epoch, criterion, test_best=0, mode='val'):
    model.eval()
    test_loss = AverageMeter('Loss', ':.4e')
    correct = AverageMeter('Acc@1', ':6.2f')  # for classification
    t0 = time.time()
    with torch.no_grad():
        for data, target, index in tqdm(loader, unit='batch'):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data, filter=args.filter)
            test_loss.update(criterion(output, target).item(), data.size(0))
            acc1 = compute_topk_accuracy(output, target, topk=(1,))
            correct.update(acc1[0].item(), data.size(0))

    log(args.logpath, 'Time for {}-Epoch-{}/{}:{:.1f}s Acc:{}, Loss:{}\n'.format('Val' if mode == 'val' else 'Test',
                                                                            epoch, args.n_epoch, time.time() - t0,
                                                                            correct.avg, test_loss.avg))

    log_value('Val/loss' if mode == 'val' else 'Test/loss', test_loss.avg, step=epoch)
    # Logging results on tensorboard
    log_value('Val/accuracy' if mode == 'val' else 'Test/accuracy', correct.avg, step=epoch)
    # Save checkpoint.
    acc = correct.avg
    if acc > test_best and mode == 'val':
        test_best = acc
        checkpoint(acc, epoch, model, args.log_dir)
    return test_best, test_loss.avg, correct.avg