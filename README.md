<p align="center">

  <h1 align="center">Bridging Diffusion Models and 3D Representations:

A 3D Consistent Super-Resolution Framework</h1>
  <p align="center">
    <a href="https://jamie725.github.io/website/">Yi-Ting Chen</a>
    ·
    <a href="https://tinghliao.github.io/">Ting-Hsuan Liao</a>
    ·
    <a href="https://psguo.github.io/">Pengsheng Guo</a>
    ·
    <a href="https://www.alexander-schwing.de/">Alexander Schwing</a>
    ·
    <a href="https://jbhuang0604.github.io/">Jia-Bin Huang</a>

  </p>
  <h2 align="center">ICCV 2025</h2>

  <h3 align="center"><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Bridging_Diffusion_Models_and_3D_Representations_A_3D_Consistent_Super-Resolution_ICCV_2025_paper.html">Paper</a> | <a href="https://arxiv.org/abs/2508.04090">arXiv</a> | <a href="https://consistent3dsr.github.io/">Project Page</a> </h3>
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="./media/trex.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We introduce a Super Resolution (3DSR), a novel 3D Gaussian-splatting-based super-resolution framework that leverages off-the-shelf diffusion-based 2D super-resolution models. 3DSR encourages 3D consistency across views via the use of an explicit 3D Gaussian-splatting-based scene representation.
</p>
<br>

# Installation
Clone the repository and create an anaconda environment using
```
git clone git@github.com:Consistent3DSR/3DSR.git
cd 3DSR

conda create -y -n 3dsr python=3.8
conda activate 3dsr

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

# Dataset
## LLFF Dataset
Please download and unzip nerf_synthetic.zip from the [LLFF](https://bmild.github.io/llff/). 

## Mip-NeRF 360 Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and request the authors for the treehill scenes.

# Training and Evaluation
```
# single-scale training and single-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_stmt.py 

# multi-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_mtmt.py 

# single-scale training and single-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360.py 

# single-scale training and multi-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360_stmt.py 
```

# Acknowledgements
This project is built upon [MipSplatting](https://github.com/autonomousvision/mip-splatting) and [StableSR](https://github.com/IceClear/StableSR). Please follow the license of MipSplatting and StableSR. We thank all the authors for their great work and repos. 

# Citation
If you find our code or paper useful, please cite
```bibtex
@inproceedings{chen2025bridging,
  title={Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework},
  author={Chen, Yi-Ting and Liao, Ting-Hsuan and Guo, Pengsheng and Schwing, Alexander and Huang, Jia-Bin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13481--13490},
  year={2025}
}