# SRefiner
SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement


## Introduction
This is the project page of the paper. In this repo, we release the code of SRefiner on Argoverse v2 with the baseline as Forecast-MAE.
* Xiao L, Pan Z, Wang Z, et al. SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 960-969.

Paper Link: [paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_SRefiner_Soft-Braid_Attention_for_Multi-Agent_Trajectory_Refinement_ICCV_2025_paper.pdf)

![pipeline](./pipeline.png)

## Qualitative Results
* On Argoverse v2 motion forecasting dataset (multi-agent track)
![vis](./vis.png)

## Getting Started
### Install dependencies
* Create a new conda virtual env
  ```bash
  conda create --name srefiner python=3.8
  conda activate srefiner
  ```
* Install PyTorch according to your CUDA version. We recommend CUDA >= 11.1, PyTorch >= 1.8.0.
  ```bash
  conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
  ```
* Install Argoverse 1 & 2 APIs, please follow [argoverse-api](https://github.com/argoai/argoverse-api) and [av2-api](https://argoverse.github.io/user-guide/getting_started.html).
* Install other dependencies
  ```bash
  pip install scikit-image IPython tqdm ipdb tensorboard
  ```

### Play with pretrained models
Evaluate on the pretrained model
  ```bash
  cd SRefiner
  bash av2_script/fmae-av2-refine-multiagent_eval_ddp.sh
  ```

### Have a train
  ```bash
  cd SRefiner
  bash av2_script/fmae-av2-refine-multiagent-train_ddp.sh
  ```

### Acknowledgement
We would like to express sincere thanks to the authors of the following packages and tools:
* [SIMPL](https://github.com/HKUST-Aerial-Robotics/SIMPL)
* [HiVT](https://github.com/ZikangZhou/HiVT)
* [SmartRefine](https://github.com/opendilab/SmartRefine)
