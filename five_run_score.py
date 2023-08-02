# -*- coding: utf-8 -*-
# Author: jlgao HIT-SCIR
import os
import re
import numpy


def cal_avg(path,reverse=True,top3=False,choose=[]):
    histo = []
    for seed_n in os.listdir(path):
        # res_path = os.path.join(path, "%s/weights" % (seed_n))
        # print(res_path, end="\t")
        res_path = os.path.join(path,seed_n,'best_results.txt')
        if not os.path.exists(res_path):
            continue
        # 只取seed0,1,2
        if not top3:
            if seed_n not in ['seed0','seed1','seed2']: 
                continue
        elif choose:
            if seed_n not in choose:
                continue
        with open(res_path, 'r', encoding='utf-8') as f:
            last_line = f.readline()
        test_val_acc = float(last_line.strip().split("\t")[0]) # Val, Test, test_val
        test_acc = float(last_line.strip().split("\t")[-1])
        histo.append((seed_n,test_acc, test_val_acc))

    histo.sort(key=lambda x: x[1],reverse=reverse)
    # histo = histo[:5]
    histo = histo[:3]

    histo_value = [t[1] for t in histo]
    avg_score = sum(histo_value) / len(histo_value)
    test_val_value = [t[2] for t in histo]
    avg_test_val = sum(test_val_value) / len(test_val_value)
    print("%s\tTest: %.2f±%.2f\ttest_val: %.2f±%.2f" % (path, avg_score, numpy.std(histo_value),avg_test_val, numpy.std(test_val_value)))

    for seed_n, sc,test_val in histo:
        print(seed_n, sc, test_val)
    fname = 'five_run_top3_score.txt' if top3 else 'five_run_score.txt'
    with open(os.path.join(path, fname), 'w', encoding='utf-8') as f:
        f.write("%s\tTest: %.2f±%.2f\ttest_val: %.2f±%.2f\n" % (path, avg_score, numpy.std(histo_value), avg_test_val, numpy.std(test_val_value) ))
        for seed_n, sc, test_val in histo:
            f.write(f'{seed_n} Test: {sc} Test_val: {test_val}\n')


if __name__ == '__main__':
    print("===> Seed 0,1,2 avg")


    # cal_avg('nrun/SST_STGN_GCE/nr0.2', reverse=True) #q=0.7
    # cal_avg('nrun/SST_STGN_GCE/nr0.4', reverse=True)
    # cal_avg('nrun/SST_STGN_GCE/nr0.6', reverse=True)

    # cal_avg('nrun/SST_STGN/nr0.2', reverse=True)
    # cal_avg('nrun/SST_STGN/nr0.4', reverse=True)
    # cal_avg('nrun/SST_STGN/nr0.6', reverse=True)

    print("===> Five run top3")

    # cal_avg('nrun/SST_base/nr0.0', reverse=True,top3=True)
    # cal_avg('nrun/SST_base/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_base/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_base/nr0.6', reverse=True,top3=True)
    
    # cal_avg('nrun/SST_SLN/nr0.2', reverse=True,top3=True,choose=['seed0','seed1','seed4'])
    # cal_avg('nrun/SST_SLN/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_SLN/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GCE/nr0.2', reverse=True,top3=True) #q=0.7
    # cal_avg('nrun/SST_GCE/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE/nr0.6', reverse=True,top3=True)

    cal_avg('/home/mjtang/wtt/Extend_T/nrun/cifar10s/0.05_0.15/ours_instance_dwt_J=9_enh_red_lam=0.5_wm=25_del=0.2_eps=0.4_eta=0.2_inc=0.01', reverse=True,top3=True) #q=0.7
    cal_avg('/home/mjtang/wtt/Extend_T/nrun/cifar10s/0.15_0.05/ours_instance_dwt_J=9_enh_red_lam=0.3_wm=30_del=0.2_eps=0.4_eta=0.1_inc=0.02', reverse=True,top3=True)
    cal_avg('/home/mjtang/wtt/Extend_T/nrun/cifar10s/0.1_0.3/ours_instance_dwt_J=9_enh_red_lam=0.3_wm=20_del=0.05_eps=0.4_eta=0.3_inc=0.02', reverse=True,top3=True)
    cal_avg('/home/mjtang/wtt/Extend_T/nrun/cifar10s/0.2_0.2/ours_instance_dwt_J=9_enh_red_lam=0.3_wm=25_del=0.1_eps=0.30000000000000004_eta=0.2_inc=0.02', reverse=True,top3=True)
    cal_avg('/home/mjtang/wtt/Extend_T/nrun/cifar10s/0.3_0.1/ours_instance_dwt_J=9_enh_red_lam=0.3_wm=25_del=0.05_eps=0.30000000000000004_eta=0.1_inc=0.01', reverse=True,top3=True)
    cal_avg('/home/mjtang/wtt/Extend_T/nrun/cifar10s/0.45_0.15/ours_instance_dwt_J=9_enh_red_lam=0.3_wm=25_del=0.05_eps=0.4_eta=0.3_inc=0.02', reverse=True,top3=True)

    # cal_avg('nrun/SST_STGN/nr0.2', reverse=True,top3=True, choose=['seed0','seed2','seed3'])
    # cal_avg('nrun/SST_STGN/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_STGN/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GNMO/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GNMO/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GNMO/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GNMP/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GNMP/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GNMP/nr0.6', reverse=True,top3=True)
    
    # cal_avg('nrun/SST_SLN-sigma0.1/nr0.2',top3=True)
    # cal_avg('nrun/SST_SLN-sigma0.2/nr0.2',top3=True)
    # cal_avg('nrun/SST_SLN-sigma0.5/nr0.2',top3=True)
    # cal_avg('nrun/SST_SLN-sigma1/nr0.2',top3=True,choose=['seed0','seed1','seed4'])

    # print("===> GCE 0,1,2 AVG")

    # cal_avg('nrun/SST_GCE-q0.4/nr0.2', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.4/nr0.4', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.4/nr0.6', reverse=True)

    # cal_avg('nrun/SST_GCE-q0.5/nr0.2', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.5/nr0.4', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.5/nr0.6', reverse=True)

    # cal_avg('nrun/SST_GCE/nr0.2', reverse=True) #q=0.7
    # cal_avg('nrun/SST_GCE/nr0.4', reverse=True)
    # cal_avg('nrun/SST_GCE/nr0.6', reverse=True)

    # cal_avg('nrun/SST_GCE-q0.9/nr0.2', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.9/nr0.4', reverse=True)
    # cal_avg('nrun/SST_GCE-q0.9/nr0.6', reverse=True)

    # print("===> GCE top3 AVG")

    # cal_avg('nrun/SST_GCE-q0.4/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.4/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.4/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GCE-q0.5/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.5/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.5/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_GCE/nr0.2', reverse=True, top3=True) #q=0.7
    # cal_avg('nrun/SST_GCE/nr0.4', reverse=True, top3=True)
    # cal_avg('nrun/SST_GCE/nr0.6', reverse=True, top3=True)

    # cal_avg('nrun/SST_GCE-q0.9/nr0.2', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.9/nr0.4', reverse=True,top3=True)
    # cal_avg('nrun/SST_GCE-q0.9/nr0.6', reverse=True,top3=True)

    # cal_avg('nrun/SST_STGN_GCE/nr0.2', reverse=True)
    # cal_avg('nrun/SST_STGN_GCE/nr0.4', reverse=True)
    # cal_avg('nrun/SST_STGN_GCE/nr0.6', reverse=True)

    # cal_avg('ablation/0/SST_STGN/nr0.2', reverse=True)
    # cal_avg('ablation/0/SST_STGN/nr0.4', reverse=True)
    # cal_avg('ablation/0/SST_STGN/nr0.6', reverse=True)

    # cal_avg('ablation/1/SST_STGN/nr0.2', reverse=True)
    # cal_avg('ablation/1/SST_STGN/nr0.4', reverse=True)
    # cal_avg('ablation/1/SST_STGN/nr0.6', reverse=True)



