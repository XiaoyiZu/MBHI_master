In this repository, we release the code and data for training a PointNet++ classification network on point clouds sampled from 3d shapes, as well as for training a part segmentation network on the ShapeNet Part dataset.
The code has been tested with Python 3.9.20, 2.0.0+cu118 on Window11.
Installation: pip install -r requirements.txt
Usage: To train the model to classify point clouds sampled from the given dataset (Zu, X. Regional characteristics of heritage houses, Tibetan houses in the northeastern region of Aba prefecture, classification datasets. Mendeley Data 1. DOI:10.17632/jps788rr2c.1; Zu, X. Regional characteristics of heritage houses, Tibetan houses in the northeastern region of Aba prefecture, semantic segmentation datasets. Mendeley Data, V1 (2024). DOI:10.17632/5b3rfrjxvg.1): train_PointMLP.py; 
To generate the P_Grad_CAM through pointnet2 backbone by: gradcam_pointnet2_weighted.py; through pointmlp by: predict_Saliency_PointMLP.py
