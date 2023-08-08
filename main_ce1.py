import os
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import time
import PreResNet_rours
from data import get_cifars_dataset, get_miniimagenet_dataset, get_Clothing1M_train_and_val_loader
from wiki_dataset import get_wiki_train_and_val_loader, get_wiki_model
from utils import train_ce, train_ours, evaluate, set_seed, log, generate_log_dir, checkpoint, lrt_correction_pr
from loss import CELoss, CE_OurLoss, OurLoss, Our_SupCL_loss
import json

from hyperopt import STATUS_OK

# TODO RunTimeError on nan
# torch.autograd.set_detect_anomaly(True)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help="No.")
    parser.add_argument('--d', type=str, default='output', help="description")
    parser.add_argument('--p', type=int, default=0, help="print")
    parser.add_argument('--c', type=int, default=10, help="class")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
    #parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.4)
    parser.add_argument('--noise_rate1', type=float, help='open corruption rate, should be less than 1', default=0.1)
    parser.add_argument('--noise_rate2', type=float, help='closed corruption rate, should be less than 1', default=0.1)
    parser.add_argument('--noise_type', type=str, help='[instance, pairflip, symmetric, asymmetric]', default='instance')
    parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, cnwl, or imagenet_tiny', default='cifar10s')
    parser.add_argument('--n_epoch', type=int, default=100)#100
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=300)
    parser.add_argument('--model_type', type=str, help='[ce, ours, ours_cl]', default='ours')
    parser.add_argument('--split_per', type=float, help='train and validation', default=0.9)
    parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
    parser.add_argument('--weight_decay', type=float, help='l2', default=5e-4)
    parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=128)

    #TODO: newly added by wtt, hyperparameters:[warm_up, delta, epsilon, eta], the last one/two more important?
    parser.add_argument('--path', type=str, help='path prefix', default='./')
    parser.add_argument('--restart', default=True, const=True, action='store_const',
                        help='Erase log and saved checkpoints and restart training')#False

    #for representation enhancement
    parser.add_argument('--lam', default=1.0, help="weight for representation enhancement", type=float)
    parser.add_argument('--f_type', type=str, default='enh', help='realize enh or enh_red with filter: [enh, enh_red]')
    parser.add_argument('--filter', type=str, default='dwt', help='Executing Filter or not: [dwt, dct, None]')
    parser.add_argument('--wvlname', type=str, default='haar', help='Which wavelet to use: [haar, dbN]')
    #for J's initialization, see line 171,172 in utils.py
    parser.add_argument('--J', type=int, default=9, help='Number of levels of decomposition')
    parser.add_argument('--data_len', type=int, default=512, help='Dimention of data before dwt')
    # J=11, data_len=2048 for clothing1m
    parser.add_argument('--mode', type=str, default='zero', help='Padding scheme: [zero, symmetric, reflect, periodization]')
    #for contrastive learning
    parser.add_argument('--gamma', default=1.0, help="weight for contrastive loss", type=float)
    parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')#[0.1,0.5]
    parser.add_argument('--aug_views', type=int, default=3, help='number of views for feature augmentations')#2
    #for correction
    parser.add_argument("--rollWindow", default=5, help="rolling window to calculate the confidence, make more stable, should be smaller than warm_up", type=int)
    parser.add_argument("--warm_up", default=30, help="warm-up period", type=int)
    parser.add_argument("--epsilon", default=0.3, help="for ID noise judgement, [0.3,0.9]", type=float)
    parser.add_argument("--eta", default=0.3, help="for OOD noise judgement", type=float)#0.3
    parser.add_argument("--delta", default=0.3, help="smoothing for one-hot vector, [0,0.5]", type=float)
    parser.add_argument("--inc", default=0.1, help="for increment of epsilon, delta", type=float)

    parser.add_argument('--params_path', type=str, default='') #params.json
    parser.add_argument('--out_tmp', type=str, default='') #result.json
    parser.add_argument('--nrun', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--suffix', default='', type=str, help="suffix action")
    # TODO clothing1m
    parser.add_argument('--num_per_class', default=18976, type=int)
    args = parser.parse_args()

    if args.dataset == 'clothing1m':
        # TODO batch_size=256, however the resources may be used up
        parser.set_defaults(c=14, batch_size=64, n_epoch=40, lr=0.01, J=11, data_len=2048, num_per_class=18976)
        args = parser.parse_args()
    elif args.dataset == 'wiki':
        # Train Setting
        parser.set_defaults(noise_mode='mix', data_path='/home/mjtang/wtt/NoisywikiHow/data/wikihow', c=158, batch_size=32, n_epoch=10, lr=3e-5, data_len=768, J=9)
        args = parser.parse_args()
        args.noise_rate = args.noise_rate2

    return args


def update_args(params={}):
    if args.model_type == 'ce':
        if args.filter == 'None':
            args.filter = None
        assert args.filter == None, 'Filter should not be set~'
    elif args.model_type in ['ours', 'ours_cl']:
        assert args.filter in ['dwt', 'dct'], 'Filter should be set~'

    # update args according to params
    for key in params:
        if params[key] is not None:
            setattr(args, key, params[key])

    noise_level = f"{args.noise_rate1}_{args.noise_rate2}"
    if args.nrun:
        args.result_dir = 'nrun'
    exp_name = os.path.join(args.path,
                            args.result_dir,
                            args.dataset,
                            noise_level,
                            args.suffix,
                            "{}".format(args.model_type) + (f"-{args.aug_views}" if args.model_type=='ours_cl' else ''),
                            '{}{}{}/'.format(args.noise_type,
                                                f'_epoch{args.n_epoch}_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}',
                        '' if args.model_type == 'ce' else '_{}_J={}_{}_lam={}_wm={}_del={}_eps={}_eta={}_inc={}{}'.format(args.filter,
                                                                                args.J,
                                                                                args.f_type,
                                                                                args.lam,
                                                                                args.warm_up,
                                                                                args.delta,
                                                                                args.epsilon,
                                                                                args.eta,
                                                                                args.inc,
                                                                                '' if args.model_type == 'ours' else '_gam={}_temp={}'.format(args.gamma, args.temp))),
                                                                                f"seed{args.seed}")
                            # + '{}_{}'.format(args.noise_rate1, args.noise_rate2))
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    args.exp_name = exp_name
    args.logpath = '{}/log.txt'.format(exp_name)
    args.log_dir = os.path.join(os.getcwd(), exp_name)
    generate_log_dir(args)
    log(args.logpath, 'Settings: {}\n'.format(args))

    args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    set_seed(args)
    return args

def get_criterion_and_model(args):
    print('building model...')
    if args.model_type == 'ce':
        criterion_train, criterion_val, criterion_test = CELoss(args.c, args.device), CELoss(args.c, args.device), CELoss(args.c, args.device)
    elif args.model_type == 'ours':
        criterion_train, criterion_val, criterion_test = OurLoss(args.c, args.device), CE_OurLoss(args.c, args.device), CE_OurLoss(args.c, args.device)
    else:
        criterion_train, criterion_val, criterion_test = Our_SupCL_loss(args), CE_OurLoss(args.c, args.device), CE_OurLoss(args.c, args.device)
    num_class = args.c if args.model_type=='ce' else args.c+1
    if args.dataset in ['cifar10s','cnwl']:
        net = PreResNet_rours.ResNet18(args, num_class)
    elif args.dataset in ['clothing1m']:
        net = PreResNet_rours.ResNet50(args, num_class)
    else:
        net = get_wiki_model(args, num_class)
    
    return net, criterion_train, criterion_val, criterion_test

def main(args, params={}):
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    if args.dataset == 'cnwl':
        train_loader, val_loader, test_loader = get_miniimagenet_dataset(args)
    elif args.dataset == 'wiki':
        train_loader, val_loader, test_loader, noisy_ind, clean_ind = get_wiki_train_and_val_loader(args)
        args.noisy_ind = noisy_ind
        args.clean_ind = clean_ind
    elif args.dataset == 'clothing1m':
        train_loader, val_loader, test_loader = get_Clothing1M_train_and_val_loader(args)
    else:
        train_loader, val_loader, test_loader = get_cifars_dataset(args)

    # Define models and criterion
    net, criterion_train, criterion_val, criterion_test = get_criterion_and_model(args)

    if torch.cuda.is_available():
        net.cuda()
        criterion_train = criterion_train.cuda()
        criterion_val = criterion_val.cuda()
        criterion_test = criterion_test.cuda()
        #cudnn.beachmark = True

    if args.dataset == 'wiki' and args.suffix != 'sgd':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = LambdaLR(optimizer, lambda epoch: 1.0) # Nothing to do with lr
    if args.dataset in ['cnwl', 'clothing1m']:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epoch)
    elif args.dataset in ['cifar10s']:
        scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    

    # training
    global_t0 = time.time()
    global val_best
    val_best = 0
    res_lst = []

    val_acc_list = []
    test_acc_list = []

    rollwin = args.rollWindow if args.model_type != 'ours_cl' else args.rollWindow*args.aug_views#2
    net_record = torch.zeros([rollwin, len(train_loader.dataset), args.c+1])
    delta_smooth = torch.full((len(train_loader.dataset),), args.delta)

    # TODO
    # 记录net_record
    if args.record:
        records = torch.zeros([args.n_epoch, len(train_loader.dataset), args.c+1])

    for epoch in range(0, args.n_epoch):
        if args.model_type == 'ce':
            train_loss, train_acc = train_ce(args, net, train_loader, optimizer, epoch, scheduler, criterion_train)
        elif args.model_type in ['ours', 'ours_cl'] :
            train_loss, train_acc = train_ours(args, net, train_loader, optimizer, epoch, scheduler, criterion_train, net_record, delta_smooth)

            if epoch >= args.warm_up:  # >
                # y_cor = lrt_correction_sr(args, target, output_norm)
                output_norm_avg = net_record.mean(dim=0)
                y_tilde = np.array(train_loader.dataset.train_labels).copy()
                y_cor, delta_smooth = lrt_correction_pr(args, epoch, y_tilde, output_norm_avg, delta_smooth)

                train_loader.dataset.update_corrupted_label(y_cor.cpu().numpy())
        else:
            assert False, "Check model type, which should be in [ce, ours, ours_cl]~"
        
        if args.record:
            records[epoch] = net_record[epoch % args.rollWindow]
        # validation
        val_best, val_loss, val_acc = evaluate(args, net, val_loader, epoch, criterion_val, val_best)
        # evaluate models
        _, test_loss, test_acc = evaluate(args, net, test_loader, epoch, criterion_test, mode='test')

        res_lst.append((train_acc, val_acc, test_acc, val_best, train_loss, val_loss, test_loss))
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

    if args.record:
        torch.save(records, os.path.join(args.log_dir, "record.pt"))
    # save model at the last epoch
    checkpoint(val_acc, epoch, net, args.log_dir, last=True)
    run_time = time.time() - global_t0
    # save 4 types of acc
    with open(os.path.join(args.log_dir, 'acc_loss_results.txt'), 'w') as outfile:
        outfile.write('{}'.format(res_lst))

    id = np.argmax(np.array(val_acc_list))
    id_ = np.argmax(np.array(test_acc_list))
    test_acc_max = test_acc_list[id]
    test_acc_max_ = test_acc_list[id_]
    log(args.logpath, '\nBest Acc: {}'.format(test_acc_max))
    log(args.logpath, '\nBest Acc_: {}'.format(test_acc_max_))
    log(args.logpath, '\nTotal Time: {:.1f}s.\n'.format(run_time))
    
    # TODO
    record_file = os.path.join(args.path, "results/results.txt")
    log(record_file, '{}:\n{}\n'.format(args.exp_name, test_acc_max_))

    with open(os.path.join(args.log_dir, 'best_results.txt'), 'w') as outfile:
        outfile.write('{}\t{}'.format(test_acc_max, test_acc_max_))

    loss = -test_acc_max_
    stable_acc = np.mean(test_acc_list[-5:])
    return {'loss': loss, 'best_acc': test_acc_max_, 'test_at_best': test_acc_max, 'stable_acc': stable_acc,
            'params': params, 'train_time': run_time, 'status': STATUS_OK}


if __name__ == '__main__':
    # acclist = []
    # # reporting mean and std
    # for i in range(args.n):
    #     args.seed = i + 1
    #     args.output_dir = './' + args.d + '/' + str(args.noise_rate) + '/'
    #     if not os.path.exists(args.output_dir):
    #         os.system('mkdir -p %s' % (args.output_dir))
    #     if args.p == 0:
    #         f = open(args.output_dir + str(args.noise_type) + '_' + str(args.dataset) + '_' + str(args.seed) + '.txt', 'a')
    #         sys.stdout = f
    #         sys.stderr = f
    #     acc = main(args)
    #     acclist.append(acc)
    # print(np.array(acclist).mean())
    # print(np.array(acclist).std(ddof=1))

    args = init_args()

    print("load params from : ", args.params_path)
    # TODO params_path等变量
    params = json.load(open(args.params_path, 'r', encoding="utf-8")) if args.params_path !='' else {}
    if 'best' in params:
        params = params['best']
        args.nrun = True
    args = update_args(params)
    # TODO
    args.record = True
    res = main(args, params=params)
    # TODO
    if args.out_tmp:
        if 'ITERATION' in params:
            res['ITERATION'] = params['ITERATION']
        json.dump(res, open(args.out_tmp, "w+", encoding="utf-8"), ensure_ascii=False)