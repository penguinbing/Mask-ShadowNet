import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x

class SpectralConv(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size,
                stride = 1,
                padding = 0,
                dilation = 1,
                groups = 1,
                bias = True,
                padding_mode = 'zeros',
                spectral_norm: bool = True,
                gated: bool = False,
                *args):
        super().__init__()
        self.gated = gated
        if gated:
            out_channels = 2*out_channels
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        out = self.conv(input)
        if self.gated:
            out, gate = out.chunk(2, dim=1)
            out = out*torch.sigmoid(gate)
        return out

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.pow(image[:, :, :, :-1] - image[:, :, :, 1:], 2)) + \
        torch.mean(torch.pow(image[:, :, :-1, :] - image[:, :, 1:, :], 2))
    return loss



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'shadow_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70000,90000,13200], gamma=0.3)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], init_weight=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_weight:
        init_weights(net, init_type, gain=init_gain)
    return net




def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    init_weight = True

    if netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'mask':
        net = MaskShadow(in_nc=input_nc, out_nc=output_nc, ngf=ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, init_weight=init_weight)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)




# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [SpectralConv(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [SpectralConv(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out





# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)






# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_act=True):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            
            up = [uprelu, upconv]
            if use_act:
                up += [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)



def aug_boundary(img, k_size=23):
    pad_size = k_size//2
    img = F.pad(img, pad=(pad_size, pad_size, pad_size, pad_size), mode='reflect')
    return F.max_pool2d(img, k_size, 1)

def redu_boundary(img, k_size=23):
    pad_size = k_size//2
    img = F.pad(img, pad=(pad_size, pad_size, pad_size, pad_size), mode='reflect')
    return -F.max_pool2d(-img, k_size, 1)

def get_edge(img, k_size=5):
    pad_size = k_size//2
    img = F.pad(img, pad=(pad_size, pad_size, pad_size, pad_size), mode='reflect')
    img = F.avg_pool2d(img, k_size, 1)
    img[img==0] = 0
    img[img==1] = 0
    img[img!=0] = 1
    return img
    

class LinearModulator(nn.Module):
    def __init__(self, in_nc, out_nc, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_nc, out_nc, bias=bias)

    def forward(self, x):
        x = x.view(x.size()[:2])
        x = self.linear(x)
        return x.view(*x.size(), 1, 1)

class MaskAdaIN(nn.Module):
    def __init__(self, dimIn, epsilon=1e-8, use_bias = False):
        super(MaskAdaIN, self).__init__()
        # avoiding divide 0 error
        self.epsilon = epsilon
        # aligners
        self.in_meanModulator = LinearModulator(2*dimIn, dimIn, bias=use_bias)
        self.in_varModulator = LinearModulator(2*dimIn, dimIn, bias=use_bias)
        self.out_meanModulator = LinearModulator(2*dimIn, dimIn, bias=use_bias)
        self.out_varModulator = LinearModulator(2*dimIn, dimIn, bias=use_bias)

    def forward(self, x1, x2, mask):

        # x: N x C x W x H
        # resize mask to the size of current feature
        mask = F.interpolate(mask, size=x1.size()[2:], mode='nearest')
        # the complementary of shadow mask
        out_mask = 1 - mask
        x1_in, x2_in = x1*mask, x2*mask
        x1_out, x2_out = x1*out_mask, x2*out_mask


        x1_in_mask_sum = x1_in.sum(dim=(2,3), keepdim=True)
        x2_in_mask_sum = x2_in.sum(dim=(2,3), keepdim=True)
        x1_out_mask_sum = x1_out.sum(dim=(2,3), keepdim=True)
        x2_out_mask_sum = x2_out.sum(dim=(2,3), keepdim=True)

        # # of elements in shadow area 
        total_in_element = mask.sum(dim=(2,3), keepdim=True)
        # # of elemnts in non-shadow area
        total_out_element = out_mask.sum(dim=(2,3), keepdim=True)

        x1_in_mean = (x1_in_mask_sum / (total_in_element+self.epsilon))
        x2_in_mean = (x2_in_mask_sum / (total_in_element+self.epsilon))
        x1_out_mean = (x1_out_mask_sum / (total_out_element+self.epsilon))
        x2_out_mean = (x2_out_mask_sum / (total_out_element+self.epsilon))

        x1_in_var = torch.clamp((torch.pow((x1_in - x1_in_mean)*mask, 2).sum(dim=(2,3), keepdim=True)/(total_in_element+self.epsilon)), 0)
        x2_in_var = torch.clamp((torch.pow((x2_in - x2_in_mean)*mask, 2).sum(dim=(2,3), keepdim=True)/(total_in_element+self.epsilon)), 0)
        x1_out_var = torch.clamp((torch.pow((x1_out - x1_out_mean)*out_mask, 2).sum(dim=(2,3), keepdim=True)/(total_out_element+self.epsilon)), 0)
        x2_out_var = torch.clamp((torch.pow((x2_out - x2_out_mean)*out_mask, 2).sum(dim=(2,3), keepdim=True)/(total_out_element+self.epsilon)), 0)
        norm_x1_in = (x1_in - x1_in_mean)*torch.rsqrt(x1_in_var+self.epsilon)
        adaptive_in_mean = self.in_meanModulator(torch.cat([x1_in_mean, x2_in_mean], dim=1)) 
        adaptive_in_var = self.in_varModulator(torch.cat([x1_in_var, x2_in_var], dim=1))
        adaptive_out_mean = self.out_meanModulator(torch.cat([x1_out_mean, x2_out_mean], dim=1))
        adaptive_out_var = self.out_varModulator(torch.cat([x1_out_var, x2_out_var], dim=1))
        adaptive_x1_in = norm_x1_in*adaptive_in_var + adaptive_in_mean
        norm_x1_out = (x1_out - x1_out_mean)*torch.rsqrt(x1_out_var+self.epsilon)
        
        adaptive_x1_out = norm_x1_out*adaptive_out_var + adaptive_out_mean
        adaptive_x1 = adaptive_x1_in * mask + adaptive_x1_out * out_mask

        return adaptive_x1


    




class MaskShadow(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, ngf=32, norm_layer=nn.InstanceNorm2d, activation=nn.ReLU(True),
                 pad=nn.ReflectionPad2d, upsample_mode='resize', max_ngf=64, spectral_norm=None, gated=False):
        super().__init__()
        def nf(num_ch: int):
            return min(max_ngf, num_ch)
        self.conv0 = nn.Sequential(pad(2),
                                   SpectralConv(in_nc, ngf, 5, 1, 0, spectral_norm=spectral_norm, gated=gated),
                                   norm_layer(ngf),
                                   activation)
        self.conv1 = nn.Sequential(pad(1),
                                   SpectralConv(ngf, 2*ngf, 3, 2, 0, spectral_norm=spectral_norm, gated=gated),
                                   norm_layer(2*ngf),
                                   activation)
        self.conv2 = nn.Sequential(pad(1),
                                   SpectralConv(2*ngf, nf(4*ngf), 3, 2, 0, spectral_norm=spectral_norm, gated=gated),
                                   norm_layer(nf(4*ngf)),
                                   activation)
        self.conv3 = nn.Sequential(pad(1),
                                   SpectralConv(nf(4*ngf), nf(8*ngf), 3, 2, 0, spectral_norm=spectral_norm, gated=gated),
                                   norm_layer(nf(8*ngf)),
                                   activation)
        self.conv4 = nn.Sequential(pad(1),
                                   SpectralConv(nf(8*ngf), nf(8*ngf), 3, 2, 0, spectral_norm=spectral_norm, gated=gated),
                                   norm_layer(nf(8*ngf)),
                                   activation)  
        self.conv5 = nn.Sequential(pad(1),
                                   SpectralConv(nf(8*ngf), nf(8*ngf), 3, 2, 0, spectral_norm=spectral_norm, gated=gated),
                                   norm_layer(nf(8*ngf)),
                                   activation) 
        self.conv6 = nn.Sequential(pad(1),
                                   SpectralConv(nf(8*ngf), nf(8*ngf), 3, 2, 0, spectral_norm=spectral_norm, gated=gated),
                                   norm_layer(nf(8*ngf)),
                                   activation)
        # def att(in_nc: int, pad_type):
        #     return AttentionModule(in_nc, pad_type, spectral_norm=spectral_norm, gated=gated)
        # self.att5 = att(nf(8*ngf), pad)
        # self.att4 = att(nf(8*ngf), pad)
        # self.att3 = att(nf(8*ngf), pad)
        # self.att2 = att(nf(4*ngf), pad)
        # self.att1 = att(2*ngf, pad)
        if upsample_mode == 'resize':
            def up_conv(in_nc: int, out_nc: int):
                return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                     pad(1),
                                     SpectralConv(in_nc, out_nc, 3, 1, spectral_norm=spectral_norm, gated=gated))
        else:
            def up_conv(in_nc: int, out_nc: int):
                if spectral_norm:
                    return torch.nn.utils.spectral_norm(nn.ConvTranspose2d(in_nc, out_nc,
                                                        kernel_size=4, stride=2,
                                                        padding=1))
                else:
                    return nn.ConvTranspose2d(in_nc, out_nc,
                                              kernel_size=4, stride=2,
                                              padding=1) 

        self.deconv6 = nn.Sequential(up_conv(nf(8*ngf), nf(8*ngf)),
                                     norm_layer(nf(8*ngf)),
                                     activation)
        self.m_ada5 = MaskAdaIN(nf(8*ngf))
        self.deconv5 = nn.Sequential(up_conv(2*nf(8*ngf), nf(8*ngf)),
                                     norm_layer(nf(8*ngf)),
                                     activation)
        self.m_ada4 = MaskAdaIN(nf(8*ngf))
        self.deconv4 = nn.Sequential(up_conv(2*nf(8*ngf), nf(8*ngf)),
                                     norm_layer(nf(8*ngf)),
                                     activation)
        self.m_ada3 = MaskAdaIN(nf(8*ngf))
        self.deconv3 = nn.Sequential(up_conv(2*nf(8*ngf), nf(4*ngf)),
                                     norm_layer(nf(4*ngf)),
                                     activation)
        self.m_ada2 = MaskAdaIN(nf(4*ngf))
        self.deconv2 = nn.Sequential(up_conv(2*nf(4*ngf), 2*ngf),
                                     norm_layer(2*ngf),
                                     activation)
        self.m_ada1 = MaskAdaIN(2*ngf)
        self.deconv1 = nn.Sequential(up_conv(4*ngf, ngf),
                                     norm_layer(ngf),
                                     activation)
        #self.m_ada0 = MaskAdaIN(ngf)
        self.out = nn.Sequential(pad(3),
                                 SpectralConv(ngf, ngf, 7, 1, spectral_norm=True, gated=True),
                                 activation,
                                 pad(2),
                                 nn.Conv2d(ngf, out_nc, 5, 1),
                                 nn.Sigmoid())

    def forward(self, img, mask=None, *args):
        x0 = self.conv0(img)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x = self.conv6(x5)
        x = self.deconv6(x)
        #x = F.interpolate(x, size=x5.size()[2:], mode='bilinear')
        x = self.deconv5(torch.cat([x, self.m_ada5(x5, x, mask)], dim=1))
        #x = F.interpolate(x, size=x4.size()[2:], mode='bilinear')
        x = self.deconv4(torch.cat([x, self.m_ada4(x4, x, mask)], dim=1))
        #x = F.interpolate(x, size=x3.size()[2:], mode='bilinear')
        x = self.deconv3(torch.cat([x, self.m_ada3(x3, x, mask)], dim=1))
        #x = F.interpolate(x, size=x2.size()[2:], mode='bilinear')
        x = self.deconv2(torch.cat([x, self.m_ada2(x2, x, mask)], dim=1))
        #x4_renorm = F.interpolate(x4_renorm, size=x1.size()[2:], mode='bilinear')
        x = self.deconv1(torch.cat([x1, self.m_ada1(x1, x, mask)], dim=1))
        x = self.out(x)
        matte = x*(1-img)+img + 1e-4
        x = torch.div(img, matte)
        
        return {'pred': x}


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.extrator = VGG16FeatureExtractor()
        self.l1 = nn.L1Loss()
    def forward(self, x, y):
        x_out = self.extrator(x)
        y_out = self.extrator(y)
        loss = 0
        for i in range(3):
            loss += self.l1(x_out[i], y_out[i])
        return loss


