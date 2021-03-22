import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
import random
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
from PIL import ImageOps,Image
class MAdaINModel(BaseModel):
    def name(self):
        return 'Mask-ShadowNet'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='none')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G', 'G_per']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['input_img','final', 'shadow_mask', 'input_gt']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #image pixel value range
        self.range_img = (0,1)  

        print(self.netG)
        if self.isTrain:
            self.criterionPerceptual = networks.PerceptualLoss().to(self.device)
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
   
    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.input_gt = input['C'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)
        self.aug_shadow_mask = networks.aug_boundary(self.shadow_mask)
        
  
    def forward(self):
        self.final = self.netG(self.input_img, self.shadow_mask)['pred']

    def backward_G(self):
        self.loss_G_per = self.criterionPerceptual(self.final, self.input_gt)
        self.loss_G = self.loss_G_per
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()    
    
    def get_current_visuals(self):
        #print("final:", self.final.size(), "shadow:", self.shadow_mask.size(), "shadow_free:", self.shadow_free.size(), "shadow:", self.shadow.size(), "input_img:", self.input_img.size())
        nim = self.input_img.shape[0]
        all =[]
        for i in range(0,min(nim,5)):
            row=[]
            for name in self.visual_names:
                if isinstance(name, str):
                    if hasattr(self,name):
                        im = util.tensor2im(getattr(self, name).data[i:i+1,:,:,:], range_img=self.range_img)
                        row.append(im)           
            row=tuple(row)
            row = np.hstack(row)
            all.append(row)      
        all = tuple(all)
        allim = np.vstack(all)
        return OrderedDict([(self.opt.name,allim)])  
    
    def get_prediction(self,input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.aug_shadow_mask = networks.aug_boundary(self.shadow_mask)
        self.gt = input['C'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)  
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)     
        output = self.netG(self.input_img, self.shadow_mask, 0)       
        self.shadow_free = output['pred']
        RES = dict()
        RES['final']= util.tensor2im(self.shadow_free,scale =0, range_img=self.range_img)
        RES['input'] = util.tensor2im(self.input_img, scale=0, range_img=self.range_img)
        RES['gt'] = util.tensor2im(input['C'], scale=0, range_img=self.range_img)
        RES['mask'] = self.shadow_mask_3d
        return  RES
