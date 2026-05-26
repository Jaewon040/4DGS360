<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />

<h1 align="center">4DGS360: 360° Gaussian Reconstruction of Dynamic Objects<br>from a Single Video</h1>

<p align="center">
  <a href="https://jaewon040.github.io/" target="_blank">Jae Won Jang</a><sup>1</sup>,
  <a href="https://yeonjin-chang.github.io/" target="_blank">Yeonjin Chang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=xWBNuGkAAAAJ&hl=en" target="_blank">Wonsik Shin</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=SKAOLIMAAAAJ&hl=en" target="_blank">Juhwan Cho</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=h_8-1M0AAAAJ&hl=en" target="_blank">Nojun Kwak</a><sup>1</sup>
</p>
<p align="center">
  <sup>1</sup>Seoul National University
</p>

<p align="center">
  <a href="https://jaewon040.github.io/4dgs360/"><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href="https://arxiv.org/abs/2603.21618"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://drive.google.com/drive/folders/17hUxN0cnu2SdpJpA374Kxt-ri7tr-Qyp?usp=drive_link'><img src='https://img.shields.io/badge/Google%20Drive-Data-blue?logo=googledrive&logoColor=white' alt='Data'></a>
</p>

<div align="center">
  <img width="900px" src="./asset/2_method_eccv.png"/>
</div>

## 📝 TODO List

- [x] Release iPhone360 dataset
- [x] Release preprocessing code (AnchorTAP3D)
- [x] Release training code


## Installation
Please follow the instructions below to set up the environment:

```bash
# Create a new conda environment
conda create -n 4dgs360 python=3.10
conda activate 4dgs360

# Install dependencies
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
```

## Data preparation
### iPhone360 Dataset
Download the preprocessed iPhone dataset from [here](https://drive.google.com/drive/folders/17hUxN0cnu2SdpJpA374Kxt-ri7tr-Qyp?usp=drive_link) and place it under `./data/iPhone360/`.


### Custom Dataset
To train on a custom dataset, please follow the instruction provided by [Shape of Motion](https://github.com/vye16/shape-of-motion) for preprocessing. Note that in our case, the data should be formatted following the iPhone dataset structure.
Then, run AnchorTAP3D to reformat the data into the iPhone360 structure. See [./preproc/AnchorTAP3D/](./preproc/AnchorTAP3D/) and refer to its [README](./preproc/AnchorTAP3D/README.md) for detailed instructions.

## Visualization
To visualize results using an interactive viewer, first download the pretrained checkpoints, then run the following command:
```bash
python run_rendering.py --ckpt-path <path-to-ckpt>
```
If you want to visualize iPhone360 results with Train/Test camera, run the following command :
```bash
python run_rendering.py --ckpt-path <path-to-ckpt> data:iphone360 --data.data-dir <path-to-data> --data.camera_type original
```

## Training
### iPhone360 Dataset
```bash
python run_training.py \
    --work-dir ./outputs/iphone360/jacket \
    --port 8888 \
    data:iphone360 \
    --data.data-dir ./data/iphone360/jacket/ \
    --data.camera_type original 
```

### iPhone Dataset
After preprocessing on AnchorTAP3D,

```bash
python run_training.py \
    --work-dir ./output/iphone/haru-sit \
    --port 8888 \
    data:iphone360 \
    --data.data-dir ./data/iphone/haru-sit/ \
    --data.camera_type original 
```
'data:iphone360' means iphone360-like dataset structure


## Evaluation
Ensure that the checkpoint file `outputs/<dataset-name>/checkpoints/last.ckpt` is available. You can either obtain this by training the model or download the provided checkpoints.

### Render Images
Use the checkpoint to render images:
```bash
python run_evaluation.py --work-dir outputs/jacket/ --ckpt-path outputs/jacket/checkpoints/last.ckpt data:iphone360 --data.data-dir ./data/iphone360/jacket
```

### Compute Metrics
Evaluate the rendered images to compute quantitative metrics:
```bash
# For the iPhone,iPhone360 dataset
PYTHONPATH="." python scripts/evaluate_iphone360.py --data_dir ./data/iphone360/jacket --result_dir ./outputs/iphone360/jacket 



## Citation
```
@article{jang20264dgs360,
  title={4DGS360: 360 $\{$$\backslash$deg$\}$ Gaussian Reconstruction of Dynamic Objects from a Single Video},
  author={Jang, Jae Won and Chang, Yeonjin and Shin, Wonsik and Cho, Juhwan and Kwak, Nojun},
  journal={arXiv preprint arXiv:2603.21618},
  year={2026}
}
```

## Acknowledgement
Our implementation builds on [Shape of Motion](https://github.com/vye16/shape-of-motion), [HiMoR](https://github.com/pfnet-research/himor), and [TAPIP3D](https://github.com/zbw001/TAPIP3D). We thank the authors for open-sourcing their code.
