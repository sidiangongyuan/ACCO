# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib

import mmcv
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim
from mmengine import build_from_cfg
from torch.distributed import get_world_size


def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        model.load_state_dict(torch.load(file_list[0], map_location='cpu'), strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(
            os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch), map_location='cpu'), strict=False)

    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    # current_path = os.path.dirname(__file__)
    # current_path = os.path.join(current_path, '../logs')
    #
    # full_path = os.path.join(current_path, folder_name)

    full_path = os.path.join("/mnt/sdb/public/data/yangk/result/cosparse_logs",folder_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None, n_iter_per_epoch=0):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'CosineAnnealing':
        from timm.scheduler import CosineLRScheduler
        warmup_iters = lr_schedule_config['warmup_iters']
        warmup_ratio =  lr_schedule_config['warmup_ratio']
        min_lr_ratio = lr_schedule_config['min_lr_ratio']
        base_lr = lr_schedule_config['base_lr']
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=60,  # 总步骤数
            lr_min= base_lr * min_lr_ratio,  # 最小学习率
            warmup_lr_init=base_lr * warmup_ratio,  # 预热期间的初始学习率
            warmup_t=warmup_iters,  # 预热步骤数
            cycle_limit=1,
            t_in_epochs=False,
        )

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )



    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    for _ in range(last_epoch):
        scheduler.step()

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device, non_blocking=True)


def reduce_value(value, average=True, distributed=True):
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value).to('cuda')
    if distributed:
        world_size = get_world_size()
        with torch.no_grad():
            from torch.distributed import distributed_c10d as dist
            dist.all_reduce(value)  # 对不同设备之间的value求和
            if average:  # 如果需要求平均，获得多块GPU计算loss的均值
                value /= world_size
    return value