# Overview
◦ Utilized techniques of Machine Learning to accomplish the sophisticated work of identifying 6 heterogeneous apple diseases, resulting in accelerating curing and preventing pandemic  
◦ Exploited Multilabel-encoding to divide every hybrid label into 6 labels with binary code in order to explicitly assort every individual disease, resulting in a higher F1-Score  
◦ Adapted myriad Batch Sizes and Data Augmentation, applied Resnet18 and Inception_V3 in the training set, and then made the comparison, resulting in 0.76387 points and 292 places in 626 competitors (Top 46.65%) on Kaggle  
# Chapter 1 Introduction
Apple is one of the world's important temperate fruit crops. Leaf diseases pose a significant threat to the overall productivity and quality of apple orchards. Currently, the disease diagnosis process in apple orchards is based on manual inspection, which is time-consuming and costly. Although computer vision-based models have shown promise in identifying plant diseases, there are still some limitations that need to be addressed. The significant variations in visual symptoms of the same disease among different apple varieties or newly cultivated varieties are a major challenge for computer vision-based disease identification. These variations are due to differences in natural and image capture environments, such as leaf color and morphology, the age of the infected tissue, uneven image backgrounds, and different lighting during the imaging process.  
  
The Plant Pathology 2021-FGVC8 dataset contains approximately 23,000 high-quality RGB images of apple leaf diseases, including a large dataset of diseases labeled by experts. This dataset reflects real field scenarios by representing non-uniform backgrounds of leaf images taken at different maturity stages and different times of the day with various camera settings.  
  
The main goal of this project is to develop a machine learning-based model to accurately classify given leaf images in the test dataset into specific disease categories and identify various diseases from multiple disease symptoms on a single leaf image.  
  
Using deep neural networks for multi-label plant disease classification can provide a faster and more convenient way to quickly distinguish various plant disease symptoms in plant disease prevention and plant pathology. This project uses various Deep learning models to classify multi-label plant disease symptoms on the Plant Pathology 2021-FGVC8 dataset. By using multi-label encoding to convert the labels in the dataset into arrays of 1s and 0s, and using data augmentation methods such as horizontal flipping, vertical flipping, and brightness adjustment, the training accuracy is optimized.  
  
This project is divided into seven chapters. Chapter 1 is the introduction, which introduces the research motivation and purpose. Chapter 2 introduces the relevant data used for multi-label plant disease classification. Chapter 3 introduces the various neural network models and loss functions used in this project. Chapter 4 introduces the evaluation methods and metrics used in this project. Chapter 5 describes the experimental methods. Chapter 6 presents the experimental results and discussion. Chapter 7 is the conclusion, followed by references.  

# Chapter 2 Dataset
## Plant Pathology 2021-FGVC8
This project uses the training set of the Plant Pathology 2021-FGVC8 dataset to split into training and validation sets and uploads the results to the Kaggle submission platform. The unpublished test set of this competition is used for evaluation.  

# Chapter 2 Dataset
This dataset was released by FineGrainedVisualCat on Kaggle in 2021. It contains 3,651 high-quality images of various apple leaf diseases captured manually by them. The images exhibit variable lighting, angles, surfaces, and noise. There are a total of 6 disease labels (as shown in Figure 1) and 12 combinations of images (as shown in Figures 2 to 13).  
  
Figure 1. Several diseases  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/abd3523d-4e4e-42ee-8068-5713fbd54e5d)  
Figure 2. Healthy  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/638d39d2-635f-4366-a0ec-2df6805db156)  
Figure 3. Scab + Frog_eye_leaf_spot  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/7e33030e-7ed1-4c24-b2ce-6303f82321fb)  
Figure 4. Scab + Frog_eye_leaf_spot + Complex  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/91907894-ed17-4bdf-af32-76735f3b93cb)  
Figure 5. Powdery_mildew + Complex  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/568af2a0-eae4-4ab6-b4b9-b6122cfe142a)  
Figure 6. Scab  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/dc98bc96-6f60-4486-abb7-d927e8c10f2b)  
Figure 7. Frog_eye_leaf_spot  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/c1dea865-0901-410c-aae4-10e50a64799d)  
Figure 8. Frog_eye_leaf_spot + Complex  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/a23dde53-9532-474d-aaa9-759cacbcd6b4)  
Figure 9. Complex  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/b83fd7b2-4902-4a42-a1e0-9036c0021dca)  
Figure 10. Rust  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/fe4c420a-2b21-4286-b118-f9b43fbd3993)  
Figure 11. Powdery_mildew  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/8bc3055e-2d36-4ecf-901d-7898d7883cf5)  
Figure 12. Rust + Frog_eye_leaf_spot  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/9d5a6dc9-f30d-427d-96fe-77870c3aa09e)  
Figure 13. Rust + Complex  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/58240221-44aa-4946-8068-7a7cab35244f)  
  
In this project, out of the 18,632 samples, approximately 80% were used for training and 20% for validation. The test set used was the unpublished test set available on Kaggle.  

# Chapter 3 Neural Network Models  
## 3.1 Resnet18
ResNet, short for Residual Network, is a special type of neural network proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in their 2015 paper "Deep Residual Learning for Image Recognition." ResNet18, for example, consists of 18 layers of residual networks. There are also other versions such as ResNet34 and ResNet50, each comprising different numbers of layers. By stacking additional layers in deep neural networks, accuracy and performance are improved.  
  
The intuition behind adding more layers is that these layers gradually learn more complex features. For instance, in image recognition, the first layer might learn to detect edges, the second layer might learn to identify textures, and the third layer might detect objects, and so on. However, it was discovered that traditional convolutional neural network models have a maximum depth threshold. For example, in experiments, a 56-layer network showed higher error rates during training compared to a 20-layer network. This is where ResNet becomes essential.  
  
ResNet solves the problem of training deeper networks by introducing residual learning. In a traditional deep network, as the number of layers increases, the gradient can vanish or explode, making training difficult. ResNet addresses this by using residual blocks, which allow the network to learn residual functions with reference to the layer inputs, instead of learning unreferenced functions.  
  
Figure 14. Resnet18 (Paolo Napoletano, Flavio Piccoli and Raimondo Schettini, 2018)  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/d6db3e5a-7239-4f92-a7fd-c684a82f14cb)  
Figure 15. Resnet18 Architecture (Farheen Ramzan, Muhammad Usman Ghani Khan, Asim Rehmat, Sajid Iqbal, Tanzila Saba, Amjad Rehman and Zahid Mehmood, 2019)  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/3535cd0e-24ac-463c-9165-c0d57aeb2d11)  

## 3.2  Inception_v3
Inception_v3 is an image recognition model developed by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna in their paper "Rethinking the Inception Architecture for Computer Vision." This model has demonstrated an accuracy of over 78.1% on the ImageNet dataset. It represents the culmination of numerous ideas developed by several researchers over many years.  
  
Figure 16. Inception_v3 interior Architecture 1 (Christian Szegedy, VincentVanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, 2015)  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/859a1c25-1b11-43f9-8616-f1c235c9aeb0)  
Figure 17. Inception_v3 interior Architecture 2 (Christian Szegedy, VincentVanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, 2015)  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/844ec1b5-1b3a-4f38-8b32-d681dfcc9b8f)  
Figure 18. Inception_v3 interior Architecture 3 (Christian Szegedy, VincentVanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, 2015)  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/b4a28779-f43e-45aa-914c-a60257fb5fd8)  
Figure 19. Inception_v3 (Christian Szegedy, VincentVanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna, 2015)  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/ddeb750c-06e5-4a44-94a3-0b8aa7a0eded)  
Figure 20. Inception_v3 Architecture (Leyan, 2021)  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/ccb17c6a-898b-4c56-8b97-e19afec5959b)  

## 3.3  Loss function
The loss function is an evaluation algorithm that assesses how well the dataset is modeled. If the predictions are completely wrong, the loss function will output a higher number. Conversely, if the modeling is very good, the loss function will output a lower number. Among the various loss functions available for multi-label prediction, I chose BCELoss (Binary Cross Entropy Loss) for this task.  
  
Figure 21. BCELoss (Bhuvana Kundumani, 2019)  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/28a9d2a5-04da-42ae-b660-45d0262f3fa2)  
  
## 3.4  Optimizer
Optimizers are a crucial component responsible for adjusting the parameters (weights and biases) of a neural network during training in order to minimize the loss function. Optimizers play a vital role in improving the efficiency and effectiveness of the training process by guiding how the model learns from the training data. I chose Adam (Adaptive Moment Estimation) which is a popular and highly effective optimizer used in deep learning for this task.  
Adam has become a cornerstone optimizer in deep learning due to its adaptive learning rate and robust performance characteristics. Its integration in this project aims to leverage these strengths to achieve effective and efficient training of models for accurate plant disease diagnosis.  

# Chapter 4 Metric 
F1-score is one of the most important evaluation metrics in machine learning. It combines two competing metrics—Precision and Recall—to summarize the predictive performance of a model. The Confusion Matrix shows the distribution of predicted labels compared to actual labels. Negative and Positive respectively indicate whether predicted labels match actual labels, while Actually False and Actually True denote whether the actual labels are true.  
  
Figure 22. Confusion Matrix  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/d9551152-9f24-4922-848a-3b45c3f3401d)  
  
Formatically, Precision and Recall are defined as:  
$Precision = \frac{TP}{TP+FP}$  
$Recall = \frac{TP}{TP+FN}$  

F1-score is defined as:  
$F1-score = {2}\times{\frac{{Precision}\times{Recall}}{{Precision}+{Recall}}}$  

# Chapter 5 Experimental Method
## 5.1  Resnet18 Model Establishment
The reason for choosing ResNet18 instead of models with more parameters and deeper residual networks like ResNet34 or ResNet50 is because increasing model complexity does not necessarily improve training effectiveness. In fact, it can lead to poorer training outcomes, including overfitting. Therefore, I opted for ResNet18, which has fewer parameters, and I explored various modifications in the final fully connected layer.  
  
One modification involved structuring the output of the fully connected layer as shown in Figure 10. With six disease categories to classify, I designed the output structure to be a one-dimensional array with 6 outputs. Using Sigmoid activation ensures the outputs are scaled between 0 and 1.  
  
Figure 23. Fully connected layer in Resnet18  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/3c142292-c625-489a-8aa9-765099a52063)  
  
The second modification involves using the structure shown in Figure 11 for the fully connected layer. Adding Dropout helps reduce overfitting.  
  
Figure 24. Fully connected layer in Resnet18 + Dropout  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/9818088a-8e96-49df-ac9f-72c3fe5fb330)  

## 5.2  Inception_v3 Model Establishment
The first modification involves using the structure depicted in Figure 12 for the fully connected layer. With six disease categories to classify, I structured the output as a one-dimensional array with 6 outputs. Sigmoid activation ensures the outputs range between 0 and 1.  
  
Figure 25. Fully connected layer in Inception_v3  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/8fade7a3-fb51-4913-8953-68be8145b739)
  
The second modification involves using the structure shown in Figure 13 for the fully connected layer. Adding Dropout helps reduce overfitting.  
  
Figure 26. Fully connected layer in Inception_v3 + Dropout  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/7bfc8539-2d42-4b83-b44f-bcf782333fd6)

## 5.3  Multi-label Encoding  
Since the objective of this project is to identify multiple diseases from images, I have separated the labels for each image into a one-dimensional matrix with 6 values (as shown in the diagram below). This approach also facilitates more accurate calculation of the F1-score.  
  
Figure 27. Multi-label Encoding for every image  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/4fcd8710-061e-40cc-ae58-8b53ba1c72ef)  

## 5.4  Data Augmentation  
When datasets are rich and comprehensive, machine learning models tend to perform better and more accurately. Data augmentation helps improve the performance and outcomes of machine learning models by creating diverse samples during training. Augmenting the training dataset aids in preventing overfitting, ensuring that the trained model generalizes effectively to new data.  
  
I employed data augmentation techniques such as RandomResizedCrop from the albumentations library, which randomly adjusts the size of images and crops them to a fixed size. Additionally, I used functions like horizontal and vertical flipping, random rotations, and random brightness adjustments (as shown in Figure 24) to further augment the dataset.  
  
Figure 28. Data Augmentation for the training dataset  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/0e3d3d77-672f-4655-8572-27ae8f648bd3)  

# Chapter 6 Experimental Result and Discussion

This project employed the experimental methods mentioned above, training four different neural network models: ResNet18, ResNet18 + Dropout, Inception_v3, and Inception_v3 + Dropout. According to Figure Twenty-Eight, Inception_v3 exhibited the best performance, achieving the highest F1-Score of 0.9136 on the validation set.  
  
Using this model, Confusion Matrices for various diseases were generated (as shown in Figures Twenty-Nine to Thirty-Four). When the Inception_v3 model was deployed on Kaggle using its undisclosed test set, it achieved a Best Score of 0.76387 on the Leaderboard, placing it at 292nd out of 626 submissions (top 46.65%).  

Figure 29. The result from every different model  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/a8a871f1-a7b4-429c-9fcf-7f5c164d38c5)  
Figure 30. 'Rust' Confusion Matrix  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/b4e13bf7-f6cf-48fa-91d2-bff2ec104c69)  
Figure 31. 'Complex' Confusion Matrix  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/57900b77-d82b-4629-9738-30c2bf6c1d5c)  
Figure 32. 'Healthy' Confusion Matrix  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/d406948e-ec47-404e-877f-75f22bbfdeb4)  
Figure 33. 'Powdery_mildew' Confusion Matrix  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/15295134-c4e4-427e-9abc-bb22c453e1f9)  
Figure 34. 'Scab' Confusion Matrix  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/6611d149-711c-4b26-b69c-79da205d81a0)  
Figure 35. 'Frog_eye_leaf_spot' Confusion Matrix  
![image](https://github.com/Wayne0758/plant-pathology-2021-fgvc8/assets/120694819/de32c6ab-5abf-44ec-87bb-2c34d33cc0fb)  
  
# Chapter 7 Conclusion
Using deep neural networks for multi-label classification of plant symptoms provides a faster and more efficient way to distinguish various symptoms of plants, which is crucial for plant disease prevention and plant pathology. Clearly defining the objectives of different tasks in various datasets is essential.  
  
This project utilized 18,632 photos from the Plant Pathology 2021-FGVC8 dataset, training with deep neural networks ResNet18 and Inception_v3. Ultimately, achieving a score of 0.76387 on the Kaggle test set, it ranked 292nd out of 626 submissions (top 46.65%).  

# References
[1]	 	Paolo Napoletano, Flavio Piccoli and Raimondo Schettini. (2018) Anomaly Detection in Nanofibrous Materials by CNN-Based Self-Similarity. Sensors 2018, 18, 109 pp. 6 https://www.researchgate.net/publication/322476121_Anomaly_Detection_in_Nanofibrous_Materials_by_CNN-Based_Self-Similarity  
[2]	 Farheen Ramzan, Muhammad Usman Ghani Khan, Asim Rehmat, Sajid Iqbal, Tanzila Saba, Amjad Rehman and Zahid Mehmood. (2019) A Deep Learning Approach for Automated Diagnosis and Multi-Class Classification of Alzheimer’s Disease Stages Using Resting-State fMRI and Residual Neural Networks.
https://www.researchgate.net/publication/336642248_A_Deep_Learning_Approach_for_Automated_Diagnosis_and_Multi-Class_Classification_of_Alzheimer's_Disease_Stages_Using_Resting-State_fMRI_and_Residual_Neural_Networks  
[3]		Christian Szegedy, VincentVanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna (2015) Rethinking the Inception Architecture for Computer Vision. https://arxiv.org/abs/1512.00567  
[4]			Karam Sahoo, Ishan Dutta, Muhammad Fazal Ijaz, Marcin Wozniak, Pawan Kumar Singh (2021) TLEFuzzyNet: Fuzzy Rank-Based Ensemble of Transfer Learning Models for Emotion Recognition From Human Speeches. IEEE Access PP(99):1-1 https://www.researchgate.net/publication/357036500_TLEFuzzyNet_Fuzzy_Rank-Based_Ensemble_of_Transfer_Learning_Models_for_Emotion_Recognition_From_Human_Speeches  
[5]	 李馨尹(2020) Inception 系列 — InceptionV2, InceptionV3
https://medium.com/ching-i/inception-%E7%B3%BB%E5%88%97-inceptionv2-inceptionv3-93cd42054d23  
[6]	 Leyan(2021) Inception-v3–1st Runner Up (Image Classification) in ILSVRC 2015
https://medium.com/image-processing-and-ml-note/inception-v3-1a5f153e95d9  
[7]	 Bhuvana Kundumani (2019) Simple Neural Network with BCELoss for Binary classification for a custom Dataset. https://medium.com/analytics-vidhya/simple-neural-network-with-bceloss-for-binary-classification-for-a-custom-dataset-8d5c69ffffee  
[8]	Zeya LT (2021) Essential Things You Need to Know About F1-Score. https://towardsdatascience.com/essential-things-you-need-to-know-about-f1-score-dbd973bf1a3https://towardsdatascience.com/essential-things-you-need-to-know-about-f1-score-dbd973bf1a3  





