```
python scripts/extract_subimages.py --input \Pixiv\valid --output \Pixiv\valid_crop --crop_size 400 --step 200
python scripts/generate_meta_info.py --input \Pixiv\valid_crop --root \Pixiv --meta_info \Pixiv\meta_info\valid.txt

python scripts/extract_subimages.py --input \Pixiv\train --output \Pixiv\train_crop --crop_size 400 --step 200
python scripts/generate_meta_info.py --input \Pixiv\train_crop --root \Pixiv --meta_info \Pixiv\meta_info\train.txt

python scripts/generate_pair.py -opt options/valid_animenet_x2plus.yml

python scripts/generate_meta_info_pairdata.py --input \Pixiv\valid_pair_gt \Pixiv\valid_pair_lq --meta_info \Pixiv\meta_info\valid_pair.txt


python inference_anime.py --input inputs

```