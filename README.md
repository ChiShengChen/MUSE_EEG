# MUSE_EEG
[![arXiv](https://img.shields.io/badge/arXiv-2406.16910-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2406.16910)  
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
<!-- ![MUSE_top1](https://github.com/ChiShengChen/MUSE_EEG/assets/22126443/42c02c49-9f00-4729-89d9-8235b6051a41) -->
<!-- ![MUSE_top5](https://github.com/ChiShengChen/MUSE_EEG/assets/22126443/b4f458c2-4003-4ed5-9cd8-91731e4e8a59) -->

<img width="1075" alt="image" src="https://github.com/user-attachments/assets/61cf51a3-078c-437e-a01b-3ed4e33ed597">  






## Datasets
many thanks for sharing good datasets!
1. [Things-EEG2](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)

The data we use is "Raw EEG data" in [here](https://osf.io/3jk45/).  
<img width="523" alt="image" src="https://github.com/user-attachments/assets/2a64602f-7c29-434e-a5d7-3bcbb50cb1df">  



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
   - `python preprocessing.py` for each subject (run by per subject), note that need to modified the default `parser.add_argument('--sub', default=<Your_Subject_Want_to_Preprocessing>, type=int)`.
   - The output files will svaed in `./Data/Things-EEG2/Preprocessed_data_250Hz/`.

2. get the center images of each test condition (for testing, contrast with EEG features)
   - get images from original Things dataset but discard the images used in EEG test sessions.
  
## Image features from pre-trained models
### Script path
- `./clipvit_feature_extraction/`
### Data path (follow the original dataset setting)
- raw image: `./Data/Things-EEG2/Image_set/image_set/`
- preprocessed eeg data: `./Data/Things-EEG2/Preprocessed_data_250Hz/`
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

## Folder Structure

<img width="465" alt="image" src="https://github.com/user-attachments/assets/9291e2ba-bdc6-449f-96a9-42348de0f025" />


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ChiShengChen/MUSE_EEG&type=Date)](https://star-history.com/#ChiShengChen/MUSE_EEG&Date)

## Reference
The code is modified based on [NICE_EEG](https://github.com/eeyhsong/NICE-EEG).

## Citation
Hope this code is helpful. I would appreciate you citing us in your paper, and the github.

```
@article{chen2024mind,
  title={Mind's Eye: Image Recognition by EEG via Multimodal Similarity-Keeping Contrastive Learning},
  author={Chen, Chi-Sheng and Wei, Chun-Shu},
  journal={arXiv preprint arXiv:2406.16910},
  year={2024}
}

```
