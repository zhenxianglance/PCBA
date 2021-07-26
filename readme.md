# A Backdoor Attack against 3D Point Cloud Classifiers (ICCV 2021)

## Introduction
Thanks for your interest in our work. This repository contains a demonstration of our backdoor attack against 3D point cloud classifiers.
You will first play the attacker's role to train a surrogate classifier using an independent, small dataset.
A backdoor attack, in particular, a small set of backdoor training samples, will be created using the surrogate classifier you just trained and the small dataset used for its training.
You will then switch the role to the training party of the victim classifier.
The training set is poisoned with the backdoor training samples you just created.
Finally, you will evaluate the attack effectiveness from the oracle perspective.

## Preparation

Software
1. Install `torch` and `torchvision` according to your CUDA Version and the instructions at [PyTorch](https://pytorch.org/).
2. Install `open3d` (for visualizing the backdoor training samples) following the instructions at http://www.open3d.org/.

Dataset
1. Download ModelNet40 dataset and save to directory `./modelnet40`.
2. May need to customize `./dataset/dataset.py` in adaption to data format other than `.h5`.

Model
1. Current model is a PointNet following the implementation at https://github.com/fxia22/pointnet.pytorch.
2. The model for the surrogate classifier need not to be the same as for the victim classifier.

## Procedure
1. Train a surrogate classifier:
    ```python
    python train_surrogate.py
    ```

2. Create backdoor training samples:
    ```python
    python attack_crafting.py
    python attack_visdualization.py
    ```

3. Training of the victim classifier:
    ```python
    python train_attacked.py
    ```

4. Evaluation of the attack:
    ```python
    python attack_evaluation.py
    ```

## Citation
If you find our work useful in your research, please consider citing:

	@InProceedings{xiang2021PCBA,
	  title={A Backdoor Attack against 3D Point Cloud Classifiers},
	  author={Xiang, Zhen and Miller, David J and Chen, Siheng and Li, Xi and Kesidis, George},
	  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	  year={2021}
	}