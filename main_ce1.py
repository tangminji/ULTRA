import os
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import time
import PreResNet_rours
from data import get_cifars_dataset
from utils import train_ce, train_ours, evaluate, set_seed, log, generate_log_dir, checkpoint, lrt_correction_pr
from loss import CELoss, CE_OurLoss, OurLoss, Our_SupCL_loss

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
parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, or imagenet_tiny', default='cifar10s')
parser.add_argument('--n_epoch', type=int, default=100)#100
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=300)
parser.add_argument('--model_type', type=str, help='[ce, ours, ours_cl]', default='ours_cl')
parser.add_argument('--split_per', type=float, help='train and validation', default=0.9)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--weight_decay', type=float, help='l2', default=5e-4)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=128)

#TODO: newly added by wtt, hyperparameters:[warm_up, delta, epsilon, eta], the last one/two more important?
parser.add_argument('--path', type=str, help='path prefix', default='/users6/ttwu/script/Unified_LNL/Extend_T')
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
parser.add_argument('--mode', type=str, default='zero', help='Padding scheme: [zero, symmetric, reflect, periodization]')
#for contrastive learning
parser.add_argument('--gamma', default=1.0, help="weight for contrastive loss", type=float)
parser.add_argument('--temp', type=float, default=0.1, help='temperature for loss function')#[0.1,0.5]
#for correction
parser.add_argument("--rollWindow", default=5, help="rolling window to calculate the confidence, make more stable, should be smaller than warm_up", type=int)
parser.add_argument("--warm_up", default=10, help="warm-up period", type=int)#8
parser.add_argument("--epsilon", default=0.3, help="for ID noise judgement, [0.3,0.9]", type=float)
parser.add_argument("--eta", default=0.3, help="for OOD noise judgement", type=float)#0.3
parser.add_argument("--delta", default=0.3, help="smoothing for one-hot vector, [0,0.5]", type=float)
parser.add_argument("--inc", default=0.1, help="for increment of epsilon, delta", type=float)

args = parser.parse_args()

if args.model_type == 'ce':
    if args.filter == 'None':
        args.filter = None
    assert args.filter == None, 'Filter should not be set~'
elif args.model_type in ['ours', 'ours_cl']:
    assert args.filter in ['dwt', 'dct'], 'Filter should be set~'

exp_name = os.path.join(args.path,
                        args.result_dir + '/' + args.dataset +
                        '/{}_{}{}/'.format(args.model_type, args.noise_type,
                       '' if args.model_type == 'ce' else '_{}_J={}_{}_lam={}_wm={}_del={}_eps={}_eta={}_inc={}{}'.format(args.filter,
                                                                               args.J,
                                                                            args.f_type,
                                                                            args.lam,
                                                                            args.warm_up,
                                                                            args.delta,
                                                                            args.epsilon,
                                                                            args.eta,
                                                                            args.inc,
                                                                            '' if args.model_type == 'ours' else '_gam={}_temp={}'.format(args.gamma, args.temp)))
                        + '{}_{}'.format(args.noise_rate1, args.noise_rate2))
if not os.path.exists(exp_name):
    os.makedirs(exp_name)

args.logpath = '{}/log.txt'.format(exp_name)
args.log_dir = os.path.join(os.getcwd(), exp_name)
generate_log_dir(args)
log(args.logpath, 'Settings: {}\n'.format(args))

args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available() and args.gpu is not None:
    torch.cuda.set_device(args.gpu)
set_seed(args)

def main(args):
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader, val_loader, test_loader = get_cifars_dataset(args)

    # Define models and criterion
    print('building model...')
    if args.model_type == 'ce':
        criterion_train, criterion_val, criterion_test = CELoss(args.c, args.device), CELoss(args.c, args.device), CELoss(args.c, args.device)
        net = PreResNet_rours.ResNet18(args, args.c)
    elif args.model_type == 'ours':
        criterion_train, criterion_val, criterion_test = OurLoss(args.c, args.device), CE_OurLoss(args.c, args.device), CE_OurLoss(args.c, args.device)
        net = PreResNet_rours.ResNet18(args, args.c+1)
    else:
        criterion_train, criterion_val, criterion_test = Our_SupCL_loss(args), CE_OurLoss(args.c, args.device), CE_OurLoss(args.c, args.device)
        net = PreResNet_rours.ResNet18(args, args.c + 1)

    if torch.cuda.is_available():
        net.cuda()
        criterion_train = criterion_train.cuda()
        criterion_val = criterion_val.cuda()
        criterion_test = criterion_test.cuda()
        cudnn.beachmark = True

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

    # training
    global_t0 = time.time()
    global val_best
    val_best = 0
    res_lst = []

    val_acc_list = []
    test_acc_list = []

    rollwin = args.rollWindow if args.model_type != 'ours_cl' else args.rollWindow*2
    net_record = torch.zeros([rollwin, len(train_loader.dataset), args.c+1])
    delta_smooth = torch.full((len(train_loader.dataset),), args.delta)

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

        # validation
        val_best, val_loss, val_acc = evaluate(args, net, val_loader, epoch, criterion_val, val_best)
        # evaluate models
        _, test_loss, test_acc = evaluate(args, net, test_loader, epoch, criterion_test, mode='test')

        res_lst.append((train_acc, val_acc, test_acc, val_best, train_loss, val_loss, test_loss))
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

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
    return test_acc_max


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

    main(args)