
# Pest-Disease-Image-Classification


<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-11557c.svg?logo=python&logoColor=white"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-0080bf.svg?logo=matplotlib&logoColor=white">
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-00acdf.svg?logo=numpy&logoColor=white"></a>
<a href="#"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-55d0ff.svg?logo=TensorFlow&logoColor=white"></a>
<a href="#"><img alt="Keras" src="https://img.shields.io/badge/Keras-7ce8ff.svg?logo=Keras&logoColor=white"></a>
<!--<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-00acdf.svg?logo=pandas&logoColor=white"></a>-->
<br />
Deep Learning based Pest Disease Classification by Training CNN model. 
<br />
<!-- by **[Pratik Sahu](https://www.linkedin.com/in/abcd123/)** -->
Apr 2023

## Project Description
Farmers incur drastic economic losses every year due to the various disease that can happen to their crop. Using Deep learning based pest disease image classification can help in the following ways:

- Early detection: By using image classification algorithms, farmers can detect pest and disease outbreaks early on. This can help them take measures to control the spread of the disease, reduce crop damage and improve yield.

- Reduced costs: Traditional methods of detecting pest and diseases involve manual inspection of crops, which can be time-consuming and expensive. Image classification can help reduce these costs by automating the process and making it more efficient.

- Better crop management: By identifying specific pests and diseases affecting their crops, farmers can take targeted measures to control them. This can help reduce the amount of pesticides and other chemicals used, leading to more sustainable and environmentally friendly farming practices.

- Improved food security: By improving crop yields and reducing losses due to pest and diseases, image classification can help improve food security, particularly in developing countries where agriculture is a major source of livelihood.

Thus by using AI based tools, pest disease image classification has the potential to revolutionize farming practices, improve yields, reduce costs and improve food security.
  
  
## Project Goal
In this project we will build a **deep learning** based image classification model using **convolutional neural network** to classify the pest occured in plants like Potato, Tomato, Bell pepper. 
  
 ## Setup for Python:

1. Install Python ([Setup instructions](https://wiki.python.org/moin/BeginnersGuide))

2. Install dependent packages

```
pip3 install -r training/requirements.txt
```

## Process Overview
#### :one:   Data Acquisition
You can download the data from [kaggle](https://www.kaggle.com/arjuntejaswi/plant-village).

#### :two:   Data Preparation

- **tf dataset**
   + We will use image_dataset_from_directory api to load all images in tensorflow dataset: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
   + Few Sample Input Images with their labels are shown below
   ![alt text](https://github.com/pratikscodes/Pest-Disease-Image-Classification/blob/main/training/input_data.png?raw=true)
  

- **Resize & Scale**
  + Before we feed our images to network, we should be resizing it to the desired size. Moreover, to improve model performance, we should normalize the image pixel value (keeping them in range 0 and 1 by dividing by 256). This should happen while training as well as inference. Hence we can add that as a layer in our Sequential Model. 

- **Data augmentation**
  + You may ask Why do we need to resize (256,256) image to again (256,256)? We don't need to but this will be useful when we are done with the training and start using the model for predictions. 
  Even if someone inputs an image that is not (256,256) and this layer will resize it.

<details>
<summary> Data Splitting </summary>

- Create function `get_dataset_partitions_tf()` to split data into **train, validate, test**

- Test prepare function

- Check the size of each dataset
     ```sh
     len(train_ds), len(val_ds), len(test_ds)
     ```
- Call the function, and cache the 3 datasets 
     ```sh
    train_ds = train.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_ds = validate.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    test_ds = test.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
     ```
</details>

#### :three:   Building the CNN Model
 In this section, we perform the following tasks:
  
- Define the architecture of the Convolutional Neural Network (CNN).
  
- Visualize and save the architecture of the CNN using the visualkeras library.

- Build the model on the training dataset and evaluate it on the train and validation datasets.

- Compile the model using the adam optimizer and SparseCategoricalCrossentropy as the loss function.
  + **SparseCategoricalCrossentropy** is a good choice for CNN because it is designed to handle multi-class classification problems where the target class is represented by a single integer. 
  In other words, it is useful when the labels are not one-hot encoded.

- Fit the model on the train dataset and evaluate it on the test dataset based on accuracy.

- Plot the training and validation accuracy and loss for each epoch.
  
- Create a function for Model Inference to return the predicted_class and confidence value.

- Make predictions on the test dataset and save the model.

- Modify the neural network architecture and optimizer as needed, repeating the above steps to generate new models and save them.


## Conclusion
Based on the evaluation results, the developed neural network model demonstrates a remarkable performance with an accuracy of 98% on the test dataset. This level of accuracy is promising and suggests that the model can generalize well on unseen data. Therefore, the model can be considered reliable and useful for accurately classifying images of pest and disease in crops. The success of this project highlights the potential of deep learning techniques in the field of agriculture, which can aid in early detection and prevention of crop damage caused by pests and diseases.

![alt text](https://github.com/pratikscodes/Pest-Disease-Image-Classification/blob/main/training/predictions.png?raw=true)
