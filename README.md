![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.10](https://img.shields.io/badge/PyTorch->=1.10-blue.svg)

# Spectral Aware Softmax for Visible-Infrared Person Re-Identification
The official repository for Spectral Aware Softmax for Visible-Infrared Person Re-Identification [[pdf]](https://arxiv.org/pdf/2302.01512.pdf)

## Training

We utilize 1 3090 GPU for training and you can train the SA-Softmax with:

```bash
python train_sas.py --gpu 'your device id' --dataset 'sysu or regdb'
```
## Citation
Please kindly cite this paper in your publications if it helps your research:
```bash
@article{tan2023spectral,
  title={Spectral Aware Softmax for Visible-Infrared Person Re-Identification},
  author={Tan, Lei and Dai, Pingyang and Ye, Qixiang and Xu, Mingliang and Wu, Yongjian and Ji, Rongrong},
  journal={arXiv preprint arXiv:2302.01512},
  year={2023}
}
```

## Acknowledgement
Our code is based on [Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)[1, 2]

## References
[1] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C., Hoi. Deep learning for person re-identification: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.

[2] M. Ye, X. Lan, Z. Wang, and P. C. Yuen. Bi-directional Center-Constrained Top-Ranking for Visible Thermal Person Re-Identification. IEEE Transactions on Information Forensics and Security (TIFS), 2019.

## Contact

If you have any question, please feel free to contact us. E-mail: [tanlei@stu.xmu.edu.cn](mailto:tanlei@stu.xmu.edu.cn)
