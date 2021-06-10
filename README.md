# Face Mask Classification via Deep Learning

In the midst of the COVID-19 pandemic, the CDC recommends everyone face masks when in public to prevent spread of the virus. Many states like California have mandated mask usage in “public and workplace settings where there is high risk of exposure”. Hence, it is important to identify individuals who are not or incorrectly wearing masks to reduce viral spread and protect those at higher risk. Our project solves this through a deep learning approach that can classify a face in one of three classes: masked, unmasked, masked incorrectly.

### Python Notebooks
* **pytorch-fasterrcnn-train.ipynb**: train face detector model
* **pytorch-fasterrcnn-test.ipynb**: load face detector model and run on test images
* **FaceMaskDetection_Custom_and_Resnet.ipynb**: train and test Resnet 18, 34, and custom CNN model for face mask classification
* **FaceMaskDetection_DenseNet.ipynb**: train and test Densenet 121, 161, 201 models for face mask classification
* **FaceMaskDetection_VGG.ipynb**: train and test VGG-16, VGG-19 models for face mask classification

### How to Run
1. Download Face Mask Dataset from Kaggle:
https://www.kaggle.com/andrewmvd/face-mask-detection
Place the extracted dataset project's home directory as follows:
./input/Face_Mask_Dataset/annotations
./input/Face_Mask_Dataset/images
2. Download Face Mask 12K Dataset from Kaggle:
https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
Place the extracted dataset project's home directory as follows:
./input/Face_Mask_Dataset_12K/Test
./input/Face_Mask_Dataset_12K/Train
./input/Face_Mask_Dataset_12K/Validation
3. Run pytorch-fasterrcnn-train.ipynb to train face detector model
Model will be saved to:
./saved_models/rcnn_model.pt
* Optional: Run pytorch-fasterrcnn-test.ipynb to load face detector model and test on images
4. Run FaceMaskDetection_VGG.ipynb to train VGG models and visualize loss
5. Run FaceMaskDetection_DenseNet.ipynb to train DenseNet models and visualize loss
6. Run FaceMaskDetection_Custom_and_Resnet.ipynb to train Resnet/Custom CNN models and visualize loss
* All models will be saved to ./saved_models as .pt
