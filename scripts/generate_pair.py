import cv2.cv2 as cv2
from basicsr import create_train_val_dataloader, osp, DiffJPEG, USMSharp
import os
import logging
from basicsr.utils import (get_env_info, get_root_logger, img_util)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from realesrgan.data.degradation import Degradation


def main():
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    opt['save_folder_gt'] = "/Pixiv/valid_pair_gt/"
    opt['save_folder_lq'] = "/Pixiv/valid_pair_lq/"

    if not os.path.exists(opt['save_folder_gt']):
        os.mkdir(opt['save_folder_gt'])
    if not os.path.exists(opt['save_folder_lq']):
        os.mkdir(opt['save_folder_lq'])

    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    degradation = Degradation(opt)
    index = 0
    extension = ".png"
    for item in iter(train_loader):
        gt_li, lq_li = degradation.degrade_data(item)

        for i in range(4):
            gt = gt_li[i]
            lq = lq_li[i]

            gt = img_util.tensor2img(gt)
            lq = img_util.tensor2img(lq)

            cv2.imwrite(
                osp.join(opt['save_folder_gt'], f's{index:03d}{extension}'), gt.astype(int),
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
            cv2.imwrite(
                osp.join(opt['save_folder_lq'], f's{index:03d}{extension}'), lq.astype(int),
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
            index += 1


if __name__ == '__main__':
    main()
