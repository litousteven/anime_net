import glob

# from .niqe import calculate_niqe
# from .psnr_ssim import calculate_psnr, calculate_ssim
import cv2.cv2 as cv2
import os

from basicsr import calculate_psnr, calculate_ssim, calculate_niqe


def main():
    results_folder = "../results_esrgan"
    gt_folder = "/Pixiv/valid_pair_gt"

    psnr_li = []
    niqe_li = []
    ssim_li = []


    img_paths = sorted(glob.glob(os.path.join(results_folder, '*')))
    for img_path in img_paths:
        img_name = os.path.basename(img_path).replace("_out","")
        gt_path = os.path.join(gt_folder, img_name)

        try:
            img = cv2.imread(img_path)
            gt_img = cv2.imread(gt_path)
            if img is None or gt_img is None:
                print(f'Img is None: {img_path}, {gt_path}')
                continue
            psnr = calculate_psnr(img, gt_img, 0)
            niqe = calculate_niqe(img, 0)
            ssim = calculate_ssim(img, gt_img, 0)

            print(psnr, niqe, ssim)

            psnr_li.append(psnr)
            niqe_li.append(niqe)
            ssim_li.append(ssim)


        except Exception as error:
            print(f'Read {img_path} {gt_path} error: {error}')

    print(f"mean psnr={sum(psnr_li)/len(psnr_li)}")
    print(f"mean niqe={sum(niqe_li) / len(niqe_li)}")
    print(f"mean ssim={sum(ssim_li)/len(ssim_li)}")


if __name__ == '__main__':
    main()
