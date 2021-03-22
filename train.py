from collections import OrderedDict
from options.train_options import TrainOptions
from options.test_options import  TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from PIL import Image
import visdom
from util.util import sdmkdir
from util import util
import time
import os

opt = TrainOptions().parse()


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = create_model(opt)
model.setup(opt)
visualizer = Visualizer(opt)

total_steps = 0




for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    model.epoch = epoch

    model.train()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.optimize_parameters()
        model.cepoch=epoch
            
        ##############Visualization block
        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch)
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_losses()
            t = (time.time() - iter_start_time) / opt.batch_size
        #    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter)/dataset_size, opt, errors)
        ###################################

    if epoch %  8 == 0:
        model.save_networks('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
