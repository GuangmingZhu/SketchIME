import os
import torch
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, adjusted_rand_score
import net
import utils
import torch_geometric as tg

class SketchModel:
    def __init__(self, opt):
        
        self.opt = opt
        self.is_train = opt.is_train
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir,  opt.class_name, opt.timestamp)
        
        self.pretrain_dir = os.path.join(opt.checkpoints_dir, opt.class_name, opt.pretrain)

        self.category_component=torch.tensor(np.load("./data/"+self.opt.class_name+"/category_component.npy")).float().cuda()
        
        #kl_loss
        self.kl_loss_func=torch.nn.KLDivLoss(reduction="batchmean",log_target=True)
        self.kl_loss_func2=torch.nn.KLDivLoss(reduction="batchmean",log_target=True)

        self.optimizer = None
        self.loss_func = None
        self.loss = None
        self.confusion = None # confusion matrix
        self.multi_confusion = None

        self.net_name = opt.net_name
        self.net = net.init_net(opt)
        self.net.train(self.is_train)
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        
        #recog_loss
        self.recog_loss_func=torch.nn.CrossEntropyLoss().to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                              lr=opt.lr, 
                                              betas=(opt.beta1, 0.999))
            self.scheduler = utils.get_scheduler(self.optimizer, opt)
        
        if not self.is_train: #or opt.continue_train:
            self.load_network(opt.which_epoch, mode='test')
            
        if self.is_train and opt.pretrain != '-':
            self.load_network(opt.which_epoch, mode='pretrain')
    
    def forward(self, x, edge_index, data,png):

        out, recog_out,out1 = self.net(x, edge_index, data,png)
        return out, recog_out,out1

    def backward(self, out, label, recog_out, recog_label,stroke_idx):
        """
        out: (B*N, C)
        label: (B*N, )
        """
        
        self.loss1 = self.loss_func(out, label)
        
        self.loss2 = self.recog_loss_func(recog_out,recog_label)
        
        
        recog_out1=recog_out.detach()
        recog_out1=torch.softmax(recog_out1,dim=1)
        category_component=self.category_component[:,:-1]
        
        recog_component=torch.mm(recog_out1,category_component)
        
        out=out[:,:-1]
        out=out.softmax(1)
        pool_stroke = torch.index_select(tg.nn.global_mean_pool(out, stroke_idx), 0, stroke_idx)
        pool_stroke=pool_stroke.view(-1,300,139)
        pool_stroke=pool_stroke.transpose(1,2)
        seg_component,_=torch.max(pool_stroke,dim=2)
        

        input=torch.log_softmax(seg_component,dim=1)
        target=torch.log_softmax(recog_component,dim=1)

        

        self.loss3=self.kl_loss_func(input,target)

        self.loss = self.loss1+100*self.loss2+self.loss3
 
        self.loss.backward()

    def step(self, data,png,epoch):
        """
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        
        
        recog_label = data.recog_label.to(self.device)
        

        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pos'] = x

        self.optimizer.zero_grad()
        out,recog_out,out1 = self.forward(x, edge_index, stroke_data,png)
        
        self.backward(out, label, recog_out, recog_label,stroke_data['stroke_idx'])
        self.optimizer.step()

    def test_time(self, data):
        """
        x: (B*N, F)
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pos'] = x

        out = self.forward(x, edge_index, stroke_data)
        
        return out
    
    def test(self, data,png, if_eval=False):
        """
        x: (B*N, F)
        """
        stroke_data= {}
        x = data.x.to(self.device).requires_grad_(self.is_train)
        label = data.y.to(self.device)
        png=png.to(self.device)
        recog_label = data.recog_label.to(self.device)
        edge_index = data.edge_index.to(self.device)
        stroke_data['stroke_idx'] = data.stroke_idx.to(self.device)
        stroke_data['batch'] = data.batch.to(self.device)
        stroke_data['edge_attr'] = data.edge_attr.to(self.device)
        stroke_data['pos'] = x
        
        out,recog_out,out1 = self.forward(x, edge_index, stroke_data,png)

        out=out.detach()
        recog_out=recog_out.detach()

        
        recog_predict_value,recog_predict=torch.topk(recog_out,1, dim=1)
        
        out=out.view(-1,300,140)
        

        recog_predict_value_k,recog_predict_k=torch.topk(recog_out,self.opt.top_k, dim=1)

        for i in range((recog_predict_value.size()[0])):
            if recog_predict_value[i].item()>self.opt.rsm_num:
                
                selected_cate=torch.index_select(self.category_component,0,recog_predict_k[i])
                
                
                selected_cate=selected_cate.transpose(1,0)
                selected_cate=torch.max(selected_cate,dim=1).values

                selected_cate=selected_cate*(1-2*self.opt.M)
                selected_cate=selected_cate+self.opt.M
                
               
                selected_cate=torch.repeat_interleave(selected_cate.unsqueeze(0),300,dim=0)

                out[i]=torch.mul(out[i],selected_cate)
                
        out=out.view(-1,140).cuda()
        recog_out=recog_out.cuda()



        predict = torch.argmax(out, dim=1).cpu().numpy()

        _,recog_predict=torch.topk(recog_out,1, dim=1)
        recog_predict=recog_predict.cpu().numpy()
        
        if (label < 0).any(): # for meaningless label
            self.loss = torch.Tensor([0])
        else:

            self.loss1 = self.loss_func(out, label)
            
            self.loss2 = self.recog_loss_func(recog_out,recog_label)
            
            self.loss=self.loss1+100*self.loss2

        return self.loss, predict,recog_predict
        
    
    def print_detail(self):
        print(self.net)

    def update_learning_rate(self):
        """
        update learning rate (called once every epoch)
        """
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)
        

    def save_network(self, epoch):
        """
        save model to disk
        """
        path = os.path.join(self.save_dir, 
                            '{}_{}.pkl'.format(self.net_name, epoch))
        
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), path)
            # print(self.net.module.cpu().state_dict().keys())
            # breakpoint()
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), path)
    
    def load_network(self, epoch, mode='test'):
        """
        load model from disk
        """
        path = os.path.join(self.save_dir if mode =='test' else self.pretrain_dir, 
                            '{}_{}.pkl'.format(self.net_name, epoch))

        
        net = self.net
        
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from {}'.format(path))
        state_dict = torch.load(path, map_location=self.device)
        
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    
