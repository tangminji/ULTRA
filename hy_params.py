# our method main.py
# tailor-made regularization
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe
from hyperopt import space_eval
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--params_path', type=str, default='params.json')
parser.add_argument('--out_tmp', type=str, default='result.json')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--sub_script', type=str, default='run_sub.sh')
parser.add_argument('--dataset', type=str, default='cifar10s')
parser.add_argument('--seed', type=int, default=0)
args, others = parser.parse_known_args()

best_acc = 0
ITERATION = 0

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main(params):
    global ITERATION
    ITERATION += 1
    params['ITERATION'] = ITERATION
    json.dump(params, open(args.params_path, 'w+', encoding="utf-8"), ensure_ascii=False)
    sig = os.system("sh %s" % args.sub_script)
    assert sig == 0
    res = json.load(open(args.out_tmp, 'r', encoding="utf-8"))
    return res

if __name__ == '__main__':

    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()

    assert args.dataset == "cifar10s"

    MAX_EVALS = 15  # TODO 设置轮次
    # TODO: times, sigma (key hyperparameters)
    # space中所有参数都需要是hp对象，否则best会缺失相应超参数值
    # python ../main_ce1.py \
    # --path ../ \
    # --model_type ours \
    # --noise_rate1 0.1 \
    # --noise_rate2 0.1 \
    # --filter dwt \
    # --n_epoch 100 \
    # --seed 1 \
    # --f_type enh_red \
    # --lam 0.5 \
    # --warm_up $warm_up \
    # --epsilon 0.3 \
    # --eta 0.3 \
    # --delta 0.1 \
    # --inc 0.01

    space = {
        # 'warm_up': 30, 
        
        # 初始标签不确定性
        'delta': hp.choice('delta', [0.05, 0.1, 0.2]),
        # delta增量
        'inc': hp.choice('inc', [0.005, 0.01, 0.02]),
        'eta': hp.choice('eta', [0.1, 0.2, 0.3]),

        # 
        # 'f_type': 'enh_red',
        'lam': hp.choice('lam', [0.3, 0.5, 0.7]),
        'epsilon': hp.quniform('epsilon', 0.2, 0.4, 0.1)
    }

    # if 'STGN' in args.exp_name:
    #     # 第三轮
    #     space = {
    #         'forget_times': hp.quniform('forget_times', 2, 5, 1), # 只有10epoch,不用设太大
    #         # 'ratio_l': hp.uniform('ratio_l', 0, 1.0), #loss vs forget的权重,0~1
    #         'ratio_l': 0.5,
    #         'avg_steps': hp.choice('avg_steps', [20]),
    #         'times': hp.choice('times', [ 10, 20, 30]),
    #         'sigma': hp.choice('sigma', [5e-3, 1e-2]),
    #         #hp.choice('sigma', [5e-4, 1e-3, 5e-3, 1e-2]),
    #     }

    best = fmin(fn=main, space=space, algo=tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(args.seed))

    print(best)
    print(bayes_trials.results)
    #TODO: use only using hp.choice
    #https://github.com/hyperopt/hyperopt/issues/284
    #https://github.com/hyperopt/hyperopt/issues/492
    print(space_eval(space, best))
    best = space_eval(space, best)
    args.log_dir = args.exp_name
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump({"best": best, "trials": bayes_trials.results},
              open(os.path.join(args.log_dir, "hy_best_params.json"), "w+", encoding="utf-8"),
              ensure_ascii=False, cls=NpEncoder)
    #os.remove(args.params_path)
    os.remove(args.out_tmp)
