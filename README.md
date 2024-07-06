# MUSE_EEG
The official repository implement of [Mind's Eye: Image Recognition by EEG via Multimodal Similarity-Keeping Contrastive Learning](https://arxiv.org/abs/2406.16910) . 
We provide new types of EEG encoders and Similarity-Keeping Contrastive Learning framework to reach the SOTA on EEG-image zero-shot classification task.
![paper_img_eeg_music_com_c](https://github.com/ChiShengChen/MUSE_EEG/assets/22126443/5099f629-4461-4455-99e2-220f6c9cedf2)

## Multimodal Similarity-Keeping ContrastivE (MUSE) Learning
![paper_img_eeg_clip_corr](https://github.com/ChiShengChen/MUSE_EEG/assets/22126443/21cf141e-39f8-4344-9006-5b22a7d6266a)
The details of the MUSE. (a.) The contrastive learning loss is calculated from EEG
encoding and image encoding. (b.)(c.) The similarity-keeping loss comes from the final similarity of
self-batch similarity of the input modal data.
## New EEG encoder series
![paper_img_model_c](https://github.com/ChiShengChen/MUSE_EEG/assets/22126443/bd584de7-87e1-486e-8344-0b88f8301fda)

## Performance
![MUSE_top1](https://github.com/ChiShengChen/MUSE_EEG/assets/22126443/42c02c49-9f00-4729-89d9-8235b6051a41)
![MUSE_top5](https://github.com/ChiShengChen/MUSE_EEG/assets/22126443/b4f458c2-4003-4ed5-9cd8-91731e4e8a59)


## Datasets
many thanks for sharing good datasets!
1. [Things-EEG2](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)
2. [Things-MEG](https://elifesciences.org/articles/82580) (updating)

## EEG pre-processing
### Script path
- `./preprocessing/`
### Data path 
- raw data: `./Data/Things-EEG2/Raw_data/`
- proprocessed eeg data: `./Data/Things-EEG2/Preprocessed_data_250Hz/`
### Steps
1. pre-processing EEG data of each subject
   - modify `preprocessing_utils.py` as you need.
     - choose channels
     - epoching
     - baseline correction
     - resample to 250 Hz
     - sort by condition
     - Multivariate Noise Normalization (z-socre is also ok)
   - `python preprocessing.py` for each subject. 

2. get the center images of each test condition (for testing, contrast with EEG features)
   - get images from original Things dataset but discard the images used in EEG test sessions.
  
## Image features from pre-trained models
### Script path
- `./clipvit_feature_extraction/`
### Data path (follow the original dataset setting)
- raw image: `./Data/Things-EEG2/Image_set/image_set/`
- preprocessed eeg data: `./Data/Things-EEG2/Preprocessed_data/`
- features of each images: `./Data/Things-EEG2/DNN_feature_maps/full_feature_maps/model/pretrained-True/`
- features been packaged: `./Data/Things-EEG2/DNN_feature_maps/pca_feature_maps/model/pretrained-True/`
- features of condition centers: `./Data/Things-EEG2/Image_set/`
### Steps
1. obtain feature maps with each pre-trained model with `obtain_feature_maps_xxx.py` (clip, vit, resnet...)
2. package all the feature maps into one .npy file with `feature_maps_xxx.py`
3. obtain feature maps of center images with `center_fea_xxx.py`
   - save feature maps of each center image into `center_all_image_xxx.npy`
   - save feature maps of each condition into `center_xxx.npy` (used in training)

## Training and testing
### Script path
- `./model/main_train.py`

## Reference
The code is modified based on [NICE_EEG](https://github.com/eeyhsong/NICE-EEG).

## Citation
Hope this code is helpful. I would appreciate you citing us in your paper, and the github.
```
@misc{chen2024muse_eeg,
  author = {Chi-Sheng Chen},
  title = {MUSE_EEG},
  year = {2024},
  version = {1.0},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ChiShengChen/MUSE_EEG}},
}

```
```
@misc{chen2024eegsk,
  title = {Mind's Eye: Image Recognition by EEG via Multimodal Similarity-Keeping Contrastive Learning},
  author = {Chen, Chi-Sheng and Wei, Chun-Shu},
  year = {2024},
  month = Jun,
  number = {arXiv:2406.16910},
  eprint = {2406.16910},
  primaryclass = {cs, eess, q-bio},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2406.16910},
  archiveprefix = {arxiv}
}
```
