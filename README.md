##Title: IMAGE CLASSIFICATION THROUGH DATA AUGMENTATION USING MACHINE LEARNING<br/>
<br/><br/>
#Dataset: Oxfordflower17<br/>
<br/><br/>
Things done:<br/>
1. AUGMENTATION<br/> 
2. FEATURE EXTRACTION<br/> 
3. TRAINING & TESTING<br/>
4. CHOOSING THE BEST CLASSIFIER<br/> 
5. PREDICTION OF FLOWERS<br/>
6. GUI<br/>
7. CLASSIFICATION IN REAL-TIME<br/>
<br/><br/>
Project idea:<br/>
We are  building an intelligent system that was trained with massive dataset of flower images.<br/>
Our system predicts the label/class of the flower of both the train and the augmented dataset with Machine Learning algorithms. Basically the data augmentation technique increases its accuracy and is far more better than the accuracy shown without using augmentation.<br/>  
<br/>
[1] AUGMENTATION:<br/>
Applyed  Augmentation Transforms-<br/>
basic transforms — Changes in angle(rotation) and lighting.<br/> 
side-on transforms — Flipping about the vertical axis<br/>
top-down transforms — Changes in angle and lighting + flipping about the horizontal axis and rotating by 10 degrees.<br/> 
Blur - We  are taking the maximum values from within a series of filters effectively removing the noise.<br/>
<br/>
[2] FEATURE EXTRACTION:<br/>
Features are the information or list of numbers that are extracted from an image.<br/> 
When deciding about the features that could quantify plants and flowers, we could possibly think of Color, Texture and Shape as the primary ones.<br/> 
This approach is likely to produce good results, but if we choose only one feature vector, as these species have many attributes in common like sunflower will be similar to daffodil in terms of color and so on. So, we need to quantify the image by combining different feature descriptors so that it describes the image more effectively.<br/>
<br/>
[3] TRAINING & TESTING:<br/>
After extracting the required features, all the global features are concatenated together.<br/>
We use k-fold cross validation technique  to avoid overfitting & split the dataset into 10 different parts.<br/>
For classification we use different classifiers such as Logistic Regression, Linear Discriminant Analysis(LDA), K Neighbors Classifier, Random Forest, Decision Tree, Gaussian NB.<br/>
<br/>
[4] CHOOSING THE BEST CLASSIFIER:<br/>
With the help of RFC algorithm, we will see how much the algorithm  is accurate in predicting the flowers.<br/>
The flowers are predicted by training both the train and train_aug dataset.<br/>
The augmented dataset is found to have high accuracy than the train dataset.<br/>
<br/>
[6] GUI:<br/> 
The GUI is developed using the GUI toolkit TKinter. <br/><br/>

[7] Classification in real-time: <br/> 
In doing the real-time classification of flowers, we have used the webcam and the mobile cam.<br/>
The interface with the mobile cam is done through IP Webcam (an android app).<br/>

