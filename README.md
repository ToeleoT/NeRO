# NeRO: Neural Road Surface Reconstruction

### [Paper](https://arxiv.org/abs/2405.10554)

![image](https://github.com/ToeleoT/NeRO/blob/main/imgs/road.png)

This project is the implementation of an MLP-based road surface reconstruction method.

## Getting Started

We run this code using Python 3.8, Pytorch 2.0.0, and CUDA 11.8.

## Dependencies

Main Python dependencies are listed below:

- Python >= 3.8
- torch >= 2.0.0 

## Installation

```
git clone https://github.com/ToeleoT/NeRO.git
cd NeRO
pip install -r requirements.txt
```

## Datasets

We mainly use the [Kitti odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) for experiments, which contains the image, lidar information, poses, and intrinsics, and then use the [Mask2Former](https://github.com/facebookresearch/Mask2Former?tab=readme-ov-file) to extract the semantic labels for experiments.

## Prepare for Datasets

Firstly, we use the [RoME](https://github.com/DRosemei/RoMe) scripts located at RoME/scripts/mask2former_infer/infer.sh to gain the semantic labels. Then, we use the code below to extract the train and semantic images, which only contain the road information.

```
python extract_road.py --img_dir image_direction --sem_dir semantic_direction
```

After that, the code extracts the intrinsics from Kitti datasets.

```
extract_intrinsics.py
```

Now, just put the training color images in the folder **train** and all the semantic labels in the folder **semantic**, and use the pose file from Kitti like "00.txt". 

To create the points resulting from road surface reconstruction, run the code below and use a pose file like "00.txt."

```
python generate_grid.py
```

Once you finish that, you can choose which type of dataset you want to use for training the network, 

- Lidar: if you want to use the lidar datasets, use the script below to extract the lidar points only on the road surface from the original files from Kitti datasets from "00_velodyne/velodyne/*.bin".

  ```
  python generate_lidar.py
  ```

- Veichle pose: if you want to use only the vehicle pose dataset, which just uses the points created by the code *<u>generate_grid.py</u>*

- SfM points: If you want to use the points from SfMs, you need to use a method like Colmaps to extract them from images, then use the same code as lidar to extract the points only on the road surface.

## Running code

Training the network with only color information, you need to use the code below:

```
python main.py --data_dir data_file --save_dir save_file --sem False
```

And if you want both semantic and color information, use the code below:

```
python main.py --data_dir data_file --save_dir save_file --sem True
```

The default method is using the hash encoding. If you want the positional encoding, just the hyperparameter like below:

```
python main.py --pe_mod 0
```

The data type is chosen using the hyperparameter below:

```
python main.py --data_type lidar
```

### Sparse label

The sparse label is using the code below:

```
python main.py --data_dir data_file --save_dir save_file --sem True --sparse_label 1 --sparse_ratio 0.1
```

### Denosing label

The denoise label code is used below:

```
python main.py --data_dir data_file --save_dir save_file --sem True --denoise 1 --denoise_ratio 0.5
```

### Other Hyperparameter

For some other hyperparameters for better network performance, you can edit the hash encodings parameters like levels and resolutions; you can change the number for the *<u>get_embedder</u>* method to fit the datasets you want.

## Acknowledgements

Thanks [nerf](https://github.com/bmild/nerf), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [semantic_nerf](https://github.com/Harry-Zhi/semantic_nerf) for providing nice and inspiring implementations of NeRF. Thanks [RoME](https://github.com/DRosemei/RoMe) for providing a great script for extracting semantic labels.

## Citation

If you found this code/work to be useful in your own research, please consider citing the following:

```
@misc{wang2024neroneuralroadsurface,
      title={NeRO: Neural Road Surface Reconstruction}, 
      author={Ruibo Wang and Song Zhang and Ping Huang and Donghai Zhang and Haoyu Chen},
      year={2024},
      eprint={2405.10554},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.10554}, 
}
```

## Contact

If you have any questions, please get in touch with howlgeass@hotmail.com or wangruibo01@saicmotor.com.

