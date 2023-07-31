import os
import ndjson
import json
import time
from options import TestOptions
from framework import SketchModel
from utils import load_data
from writer import Writer
import numpy as np
from evalTool import *
import torch

def run_eval(opt=None, model=None, loader=None, dataset='test', write_result=False):
    if opt is None:
        opt = TestOptions().parse()
    if model is None:
        model = SketchModel(opt)
    if loader is None:
        loader = load_data(opt, datasetType=dataset, permutation=opt.permutation)
    # print(len(loader)) 
    if opt.eval_way == 'align':
        predictList, lossList, recog_predictlist = eval_align_batchN(model, loader, P=opt.points_num)
    elif opt.eval_way == 'unalign':
        predictList, lossList = eval_unalign_batch1(model, loader)
    else:
        raise NotImplementedError('eval_way {} not implemented!'.format(opt.eval_way))
    # print(predictList.shape)
    testData = []
    
    
    with open(os.path.join("./data", opt.class_name,
            '{}_{}.ndjson'.format(opt.class_name, dataset)), 'r') as f:
        testData = ndjson.load(f)

    if opt.metric_way == 'wlen':
        p_metric_list, c_metric_list = eval_with_len(testData, predictList)
        
    elif opt.metric_way == 'wolen':
        p_metric_list, c_metric_list = eval_without_len(testData, predictList)
    else:
        raise NotImplementedError('metric_way {} not implemented!'.format(opt.metric_way))

    
    loss_avg = np.average(lossList)
    P_metric = np.average(p_metric_list)
    C_metric = np.average(c_metric_list)
    

    all_recog_num=len(testData)
    right_recog_num=0
    for inde,sketch1 in enumerate(testData):
        
        if int(sketch1["recog_label"]) in recog_predictlist[inde]:
            right_recog_num+=1
        

    recog_metric=float(right_recog_num)/all_recog_num


    return loss_avg, P_metric, C_metric, recog_metric




if __name__ == "__main__":
    
    
    _, P_metric, C_metric, recog_metric = run_eval(write_result=True)
    print('P_metric:{:.4}%\tC_metric:{:.4}%\tRecog_metric:{:.4}%'.format(P_metric*100, C_metric*100, recog_metric*100))


