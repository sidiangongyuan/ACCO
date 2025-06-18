# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import time
from datetime import datetime
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
import logging
import json
from icecream import ic
from torch.utils.data.distributed import DistributedSampler
from multi_gpu_utils import setup_for_distributed
from opencood.tools.train_utils import reduce_value

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    # parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
    #                     help='data generation yaml file needed ')
    parser.add_argument("--hypes_yaml", "-y", type=str,
                        help='data generation yaml file needed ',default='../hypes_yaml/opv2v/camera/'
                                                                         'opv2v_sparse4D_resnet50_multiloss_w_dir_depth_pointEncoderV6_Lidar.yaml')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')

    parser.add_argument('--new_dir',default=False, help='New dir for continued training path')

    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')

    parser.add_argument('--cuda_device', type=int, default=0)
    # 分布式设置
    parser.add_argument('--dist-gpu', default=False, type=bool,
                        help='')

    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():

    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    if 'head_args' in hypes['model']['args']:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        head_args_formatted = json.dumps(hypes['model']['args']['head_args'], indent=2)
        logging.info("Hypes: \n%s", head_args_formatted)
        logging.info("train_params: %s", hypes['train_params'])


    print('Dataset Building')
    print("batch_size:",hypes['train_params']['batch_size'])
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)
    opt.dist_gpu = opt.distributed
    if not opt.dist_gpu:
        torch.cuda.set_device(opt.cuda_device)
        print("Cuda device is %d" % (opt.cuda_device))
    if opt.dist_gpu:
        train_sampler = DistributedSampler(dataset=opencood_train_dataset, shuffle=True)
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  num_workers=4,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  pin_memory=True,
                                  prefetch_factor=2,
                                  batch_sampler=train_batch_sampler)

        val_sampler = DistributedSampler(dataset=opencood_validate_dataset, shuffle=False) 
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                pin_memory=True,
                                drop_last=False,
                                prefetch_factor=2,
                                sampler=val_sampler)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=4,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True,
                                  prefetch_factor=2)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    if opt.dist_gpu:
        hypes['optimizer']['lr'] = hypes['optimizer']['lr'] * opt.world_size
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup

    init_epoch = 0
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)
    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        print("start from " +  str(lowest_val_epoch))
        if opt.new_dir:
            print("new_dir: True")
            saved_path = train_utils.setup_train(hypes)
        # scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
    else:
        saved_path = train_utils.setup_train(hypes)

    writer = SummaryWriter(saved_path)
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = True if 'supervise_single' in hypes['train_params'] and hypes['train_params']['supervise_single'] else False
    # used to help schedule learning rate

    if opt.dist_gpu:
        print('distributed training')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Training start from {formatted_time}')

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if opt.dist_gpu:
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            if i%100 == 0:
                print("100 batch time:" + str(time.time() - start_time))
                criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()


        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    pred_results_dict = model(batch_data['ego'],'val')

                    final_loss = criterion(pred_results_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
                    if i % 100 == 0:
                        criterion.logging(epoch, i, len(val_loader), writer)
                valid_ave_loss = statistics.mean(valid_ave_loss)
                valid_ave_loss = reduce_value(valid_ave_loss, True, opt.dist_gpu)
                valid_ave_loss = valid_ave_loss.item()
                print('At epoch %d, the validation loss is %f' % (epoch,
                                                                      valid_ave_loss))
                writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        print("one epoch time:" + str(time.time() - start_time))
        scheduler.step(epoch)
    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = False
    # need to rewrite , USELESS
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python ./inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
