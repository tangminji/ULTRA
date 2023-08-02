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

def get_trials(fixed, space, MAX_EVALS):
    for k in space:
        times = len(space[k])
        break
    if times > MAX_EVALS:
        times = MAX_EVALS
    for t in range(times):
        params = {k: space[k][t] for k in space}
        params.update(fixed)
        yield params

if __name__ == '__main__':

    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()

    assert args.dataset == "cnwl"

    MAX_EVALS = 5  # TODO 设置轮次
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

    fixed = {
    }

    space = {
        'warm_up': [10, 15, 20, 25, 30],
    }
    # space = {
    #     'warm_up': [25, 30, 35],
    # }

    trials = get_trials(fixed, space, MAX_EVALS)
    all_trials = []
    best_loss = None
    best = None
    for params in trials:
        res = main(params)
        all_trials.append(res)
        loss = res['loss']
        if best_loss is None or loss<best_loss:
            best_loss = loss
            best = params

    print(best, best_loss)
    print(all_trials)

    args.log_dir = args.exp_name
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump({"best": best, "trials": all_trials},
              open(os.path.join(args.log_dir, "hy_best_params.json"), "w+", encoding="utf-8"),
              ensure_ascii=False, cls=NpEncoder)
    #os.remove(args.params_path)
    os.remove(args.out_tmp)
