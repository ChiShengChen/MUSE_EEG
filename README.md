# MUSE_EEG
The official repository implement of Mind's Eye: Image Recognition by EEG via Multimodal Similarity-Keeping Contrastive Learning (arXiv unexpected on-hold, submit/5643879).  
You can find the paper temparally from [here](https://www.linkedin.com/posts/michael-chi-sheng-chen-257359137_i-am-honored-to-share-with-you-our-new-paper-activity-7206348296766803968-UdtB?utm_source=share&utm_medium=member_desktop).
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

## Reference
The code is modified based on [NICE_EEG](https://github.com/eeyhsong/NICE-EEG).

## Citation
Hope this code is helpful. I would appreciate you citing us in your paper, arXiv now unexpected on-hold.
```
@misc{chen2024eegsk,
  title = {Mind's Eye: Image Recognition by EEG via Multimodal Similarity-Keeping Contrastive Learning},
  author = {Chen, Chi-Sheng and Wei, Chun-Shu},
  year = {2024},
  month = Jun,
  number = {arXiv:submit/5643879},
  eprint = {on-hold},
  primaryclass = {cs, eess, cv},
  publisher = {{arXiv}},
  doi = {},
  archiveprefix = {arxiv}
}
```
