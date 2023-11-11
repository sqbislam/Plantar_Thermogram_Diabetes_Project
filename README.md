# Thermal Imaging and deep learning to evaluate the presence of diabetes using machine learning and a multi-view CNN approach. 

#### The objective of this project is to experiment whether an automated image registration approach combined with region of interest feature extraction from thermal images of patients foot can accurately predict the early detection of diabetic foot ulceration (DFU). Secondly, also test a multi-view deep learning based approach which inputs both feet images to classify diabetic foot ulceration from thermal images accurately

# Deployment
The deployed application for this project can be found here [https://github.com/sqbislam/Thermal-Imaging-DiabeticFootUlceration-Detection](https://github.com/sqbislam/Thermal-Imaging-DiabeticFootUlceration-Detection)

# Recommended Requirements
This code was tested primarily on Python 3.8.12 using jupyter notebook.
The following environment is recommended.


- cytoolz==0.11.0
- flatbuffers==1.12
- fonttools==4.25.0
- keras==2.9.0
- Keras-Preprocessing==1.1.2
- matplotlib==3.5.1
- statsmodels==0.13.2
- tensorflow-cpu==2.9.1
- voxelmorph==0.2
- xgboost==0.90
- click==7.1.2
- joblib==0.17.0
- opencv-python==4.6.0.66
- lime==0.2.0.1
- shap==0.40.0


# References
The following papers and codes were the main references for the construction of the image registration model and mutli-view approach used in this study and for understanding the theory.  

1. Jin, C. et al. Predicting treatment response from longitudinal images using multi-task deep learning. Nat. Commun. 2021 121 12, 1–11 (2021).
the source code of 3D-RPNet is released under the MIT License  
Copyright (c) 2021 Heng
[https://github.com/Heng14/3D_RP-Net/blob/master/LICENSE](https://github.com/Heng14/3D_RP-Net/blob/master/LICENSE)


2. Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J. & Dalca, A. V. VoxelMorph: A Learning Framework for Deformable Medical Image Registration. IEEE Trans. Med. Imaging 38, 1788–1800 (2018).
Code of VoxelMorph image registration deep learning based model is available under the Apache License 2.0
[https://github.com/voxelmorph/voxelmorph/blob/dev/LICENSE.md](https://github.com/voxelmorph/voxelmorph/blob/dev/LICENSE.md)




