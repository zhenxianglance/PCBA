# A Backdoor Attack against 3D Point Cloud Classifiers (ICCV 2021)

## Introduction
Thanks for your interest in our work. This repository contains a demonstration of our backdoor attack against 3D point cloud classifiers.
This is the first work extending backdoor attacks which were originally proposed for images to 3D point clouds.
The attacker aims to have the victim classifier predict to a target class whenever a test example from one (or several) source classes is embedded with a backdoor pattern.
An example is shown in the figure below, where the source class, the target class, and the backdoor pattern for the backdoor attack are pedestrian, car, and a ball, respectively.
When there is no attack, the classifier correctly distinguishes pedestrians from cars. But for a classifier being attacked, a pedestrian carrying a ball will be recognized as a car, which can be catastrophic.

<p align="center">
<img src="https://github.com/zhenxianglance/PCBA/blob/main/figure1.png" width="500" />
</p>

In the following, we first give the outline of our attack. Then we will provide instructions about how to run our code.

## Method

We propose to use a small cluster of inserted points as the backdoor pattern, which can possibly be implemented in practice using physical objects (e.g. a ball).
The outline of our attack is summarized in the figure below. The attacker independently collects a small dataset containing samples from all categories and uses it to train a surrogate classifier.
This surrogate classifier is used for optimizing the spatial location and local geometry of the inserted cluster of points. Details for these two optimization problems can be found in the paper.
Then, the attacker embed the backdoor pattern to a small set of samples from the source class and label them to the target class.
These samples (with the backdoor pattern and being mislabeled) are inserted into the training set of the classifier.
After normal training process, the classifier will: 1) predict test samples from the source class to the target class when the same backdoor pattern is embedded; 2) still correctly classify test samples without the backdoor pattern.

<p align="center">
<img src="https://github.com/zhenxianglance/PCBA/blob/main/figure2.png" width="500" />
</p>

## How to use

Here, you will play the role as the attacker to launch our backdoor attack following the protocol described above. 
In addition, you will evaluate the effectiveness of the attack you created from an oracle perspective.

### Preparation

Software
1. Install `torch` and `torchvision` according to your CUDA Version and the instructions at [PyTorch](https://pytorch.org/).
2. Install `open3d` (for visualizing the backdoor training samples) following the instructions at http://www.open3d.org/.

Dataset
1. Download ModelNet40 dataset and save to directory `./modelnet40`.
2. May need to customize `./dataset/dataset.py` in adaption to data format other than `.h5`.

Model
1. Current model is a PointNet following the implementation at https://github.com/fxia22/pointnet.pytorch.
2. The model for the surrogate classifier need not to be the same as for the victim classifier.

### Procedure
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
