import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

from sklearn.metrics import f1_score, roc_auc_score, jaccard_score
from PIL import Image
import numpy as np

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def cal_pnsr(sr_img, gt_img):
    # calculate PSNR
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.
    psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)

    return psnr

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')  # config 文件
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        # resume_state = torch.load(opt['path']['resume_state'],
        #                           map_location=lambda storage, loc: storage.cuda(device_id), strict=False)
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    model = create_model(opt)
    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                avg_psnr_h = [0.0]*opt['num_image']
                avg_psnr_lr = 0.0
                avg_biterr = 0.0
                idx = 0
                for image_id, val_data in enumerate(val_loader):
                    img_dir = os.path.join(opt['path']['val_images'])
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test(image_id)

                    visuals = model.get_current_visuals()

                    t_step = visuals['SR'].shape[0]
                    idx += t_step
                    n = len(visuals['SR_h'])

                    avg_biterr += util.decoded_message_error_rate_batch(visuals['recmessage'][0], visuals['message'][0])

                    for i in range(t_step):

                        sr_img = util.tensor2img(visuals['SR'][i])  # uint8
                        sr_img_h = []
                        for j in range(n):
                            sr_img_h.append(util.tensor2img(visuals['SR_h'][j][i]))  # uint8
                        gt_img = util.tensor2img(visuals['GT'][i])  # uint8
                        lr_img = util.tensor2img(visuals['LR'][i])
                        lrgt_img = []
                        for j in range(n):
                            lrgt_img.append(util.tensor2img(visuals['LR_ref'][j][i]))

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'SR'))
                        util.save_img(sr_img, save_img_path)

                        for j in range(n):
                            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:d}_{:s}.png'.format(image_id, i, j, 'SR_h'))
                            util.save_img(sr_img_h[j], save_img_path)

                        save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'GT'))
                        util.save_img(gt_img, save_img_path)

                        save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(image_id, i, 'LR'))
                        util.save_img(lr_img, save_img_path)

                        for j in range(n):
                            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:d}_{:s}.png'.format(image_id, i, j, 'LRGT'))
                            util.save_img(lrgt_img[j], save_img_path)
                        f1_list = []
                        auc_list = []
                        iou_list = []
                        print("\n", n)
                        img_dir = "./demo/"
                        # Lấy watermark gốc và predicted (giả sử n=1)
                        wloc = lrgt_img[0]      # uint8 [H, W, 3]
                        wloc_pred = sr_img_h[0] # uint8 [H, W, 3]
                        if wloc.max() > 1.5:
                            wloc = wloc / 255.0
                            wloc_pred = wloc_pred / 255.0

                        diff = np.abs(wloc_pred - wloc)                # [H, W, 3]
                        max_diff = np.max(diff, axis=2)                # [H, W]
                        threshold = 0.2    
                        mask_bin = (max_diff >= threshold).astype(np.uint8)  # 0: nền, 1: vùng bị sửa

                        # Đổi thành 0/255 để lưu thành ảnh trắng đen dễ nhìn
                        delta_img = (mask_bin * 255).astype(np.uint8)
                        # Lưu delta
                        delta_path = os.path.join(img_dir, f'{image_id}_{i}_delta_wloc.png')
                        util.save_img(delta_img, delta_path)
                        # Lưu mask_gt
                        mask_path = f'../dataset/valAGE-Set-Mask/{image_id + i + 1:04d}.png'
                        mask_img = Image.open(mask_path).resize(wloc.shape[1::-1]).convert("L")
                        mask_gt = np.array(mask_img)  # uint8 [H, W]
                        mask_gt_path = os.path.join(img_dir, f'{image_id}_{i}_mask_gt.png')
                        # Chuyển về mask 0/1 (nếu chưa)
                        mask_pred = (delta_img > 128).astype(np.uint8)
                        mask_gt = (mask_gt > 128).astype(np.uint8)
                        y_pred = mask_pred.flatten()
                        y_true = mask_gt.flatten()

                        # Tính F1-score
                        f1 = f1_score(y_true, y_pred)
                        f1_list.append(f1)

                        # Tính AUC (cần có cả 0 và 1 trong ground truth)
                        try:
                            auc = roc_auc_score(y_true, y_pred)
                            auc_list.append(auc)
                        except ValueError:
                            auc = None  # Trường hợp mask chỉ có 1 class
                            print(f'AUC: Không tính được (1 class) cho image {image_id}_{i}')

                        # Tính IoU (Jaccard score)
                        iou = jaccard_score(y_true, y_pred)
                        iou_list.append(iou)

                        print(f'Image {image_id}_{i} - F1-score: {f1:.4f}, IoU: {iou:.4f}')
                        if auc is not None:
                            print(f'Image {image_id}_{i} - AUC: {auc:.4f}')
                        psnr = cal_pnsr(sr_img, gt_img)
                        psnr_h = []
                        for j in range(n):
                            psnr_h.append(cal_pnsr(sr_img_h[j], lrgt_img[j]))
                        psnr_lr = cal_pnsr(lr_img, gt_img)

                        avg_psnr += psnr
                        for j in range(n):
                            avg_psnr_h[j] += psnr_h[j]
                        avg_psnr_lr += psnr_lr
                avg_psnr = avg_psnr / idx
                avg_psnr_h = [psnr / idx for psnr in avg_psnr_h]
                avg_psnr_lr = avg_psnr_lr / idx
                avg_biterr = avg_biterr / idx

                # log
                res_psnr_h = ''
                for p in avg_psnr_h:
                    res_psnr_h+=('_{:.4e}'.format(p))
                logger.info('# Validation # PSNR_Cover: {:.4e}, PSNR_Secret: {:s}, PSNR_Stego: {:.4e}, Bit_err: {: .4e}'.format(avg_psnr, res_psnr_h, avg_psnr_lr, avg_biterr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> PSNR_Cover: {:.4e}, PSNR_Secret: {:s}, PSNR_Stego: {:.4e}, Bit_err: {: .4e}'.format(
                    epoch, current_step, avg_psnr, res_psnr_h, avg_psnr_lr, avg_biterr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
