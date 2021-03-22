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



test_opt = TestOptions().parse()
model = create_model(test_opt)
model.setup(test_opt)

test_data_loader = CreateDataLoader(test_opt)
test_set = test_data_loader.load_data()
test_save_path = os.path.join(test_opt.checkpoints_dir, 'test')

if not os.path.isdir(test_save_path):
    os.makedirs(test_save_path)

model.eval()
idx = 0
for i, data in enumerate(test_set):
    idx += 1
    visuals = model.get_prediction(data)
    pred = visuals['final']
    gt = visuals['gt']
    im_name = data['imname'][0].split('.')[0]
    util.save_image(gt, os.path.join(test_save_path, im_name+'_gt.png'))
    util.save_image(pred, os.path.join(test_save_path, im_name+'pred_gt.png'))

    
