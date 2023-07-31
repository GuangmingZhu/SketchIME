import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
from options import TrainOptions, TestOptions
from framework import SketchModel
from utils import load_data
from writer import Writer
from evaluate import run_eval
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
# import torchsnooper



def run_train(train_params=None, test_params=None):
    opt = TrainOptions().parse(train_params)
    testopt = TestOptions().parse(test_params)
    
    
    testopt.timestamp = opt.timestamp
    testopt.batch_size = 30
    
    # model init
    model = SketchModel(opt)
    

    writer = Writer(opt)
    
    # data load
    trainDataloader = load_data(opt, datasetType='train', permutation=opt.permutation, shuffle=opt.shuffle)
    
    testDataloader = load_data(opt, datasetType='test')

    # train epoches
    # with torchsnooper.snoop():
    ii = 0
    min_test_avgloss = 5000
    min_test_avgloss_epoch = 0

    
    acctxt=open(os.path.join(opt.expr_dir,"acc.txt"),"w")

    for epoch in range(opt.epoch):
        for i, (data,png) in enumerate(trainDataloader):
            # print(stroke_idx)
            # print(type(stroke_idx))
            # breakpoint()
            model.step(data,png,epoch)

            if ii % opt.plot_freq == 0:
                writer.plot_train_loss(model.loss, ii)
            if ii % opt.print_freq == 0:
                writer.print_train_loss(epoch, i, model.loss)

            ii += 1

        model.update_learning_rate()
        if opt.plot_weights:
            writer.plot_model_wts(model, epoch)
        
        # test
        if epoch % opt.run_test_freq == 0:
            model.save_network('latest')
            loss_avg, P_metric, C_metric, recog_metric = run_eval(
                opt=testopt,
                loader=testDataloader, 
                dataset='test',
                write_result=False)
            writer.print_test_loss(epoch, 0, loss_avg)
            writer.plot_test_loss(loss_avg, epoch)
            writer.print_eval_metric(epoch, P_metric, C_metric, recog_metric)
            writer.plot_eval_metric(epoch, P_metric, C_metric)

            w_str=str(round(loss_avg,5))+" "+str(round(P_metric,5)*100)+"%"+" "+str(round(C_metric,5)*100)+"%"+" "+str(round(recog_metric,5)*100)+"%"+"\n"
            acctxt.write(w_str)
            model.save_network(epoch)
            
            if loss_avg < min_test_avgloss:
                min_test_avgloss = loss_avg
                min_test_avgloss_epoch = epoch
                print('saving the model at the end of epoch {} with best avgLoss {}'.format(epoch, min_test_avgloss))
                model.save_network('bestloss')

    testopt.which_epoch = 'latest'
    testopt.metric_way = 'wlen'
    loss_avg, P_metric, C_metric, recog_metric = run_eval(
        opt=testopt,
        loader=testDataloader, 
        dataset='test',
        write_result=False)
    
    testopt.which_epoch = 'bestloss'
    testopt.metric_way = 'wlen'
    loss_avg_2, P_metric_2, C_metric_2,recog_metric2 = run_eval(
        opt=testopt,
        loader=testDataloader, 
        dataset='test',
        write_result=False)

    record_list = {
        'p_metric': round(P_metric*100, 2),
        'c_metric': round(C_metric*100, 2),
        'loss_avg': round(loss_avg, 4),
        'recog_metric': round(recog_metric*100,2),
        'best_epoch': min_test_avgloss_epoch,
        'p_metric_2': round(P_metric_2*100, 2),
        'c_metric_2': round(C_metric_2*100, 2),
        'loss_avg_2': round(loss_avg_2, 4),
        'recog_metric2': round(recog_metric2*100,2)
    }
    writer.train_record(record_list=record_list)
    writer.close()
    return record_list, opt.timestamp

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    record_list, _ = run_train()
    print(record_list)
