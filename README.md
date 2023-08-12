# Handwriting OCR - Image Processing using CNN

## Introduction

This project was completed in April 2023 by myself and 3 other students at the University of Waterloo for our Machine Learning course (MSCI 446). My role was primarily to work on importing, cleaning, and preparing the input images. I also had a part in designing the neural network and calculating the accuracy of the final model.


### Summary of Results

The final model features 2 RNN layers, 3 CNN layers, and an Adam optimizer run over 10 epochs.

It is able to classify exact words 27% of the time, but has an edit distance of 0.44. This means that, on average, each incorrect word is only about 1 letter off.

The total runtime of the model took 22 minutes and 43 seconds when run on a RTX 3070 GPU.

**Final model label predictions compared to original image:**
![Results](https://github.com/yzzy2go/HandwritingOCR/assets/52092038/7c2132ef-fe57-425a-b4d8-f1b7fa086e16)

## Development

### Technology Used

* Tensorflow
* Keras
* NumPy
* Jupyter Notebook
* Convolutional Neural Networks (CNN)

### Development Environment

The model was created in Jupyter Notebook. This environment allowed for seamless experimentation, code documentation, and real-time visualization that allowed us to iteratively refine the AI model.

CUDA was employed to harness GPU acceleration, leveraging the power of an RTX 3070 graphics card for enhanced computational performance.

### Dataset

The IAM Handwriting Database (version 3.0) was used as the dataset source. 115,320 PNG images of isolated and labeled words were chosen to be the starting dataset.

[Download the IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)

**Sample images from the IAM Database:**
![SampleIAMWords](https://github.com/yzzy2go/HandwritingOCR/assets/52092038/db09f643-d57e-4f6f-b2d9-d5ea3d921eda)

The tags associated with these images in the annotation file will allow the accuracy of the model to be determined, as their correct label has already been manually verified. In the image below, the first set of alphanumeric characters indicates the image file name, while the last word in the line is what the image says.

**Sample rows from annotation file:**
![AnnotationSamples](https://github.com/yzzy2go/HandwritingOCR/assets/52092038/49a13786-44e1-4a0d-aa5c-182d4f47a974)

The dataset is split into testing, training, and validation sets - with 86810, 4823, and 4823 unique images, respectively.

### Data Manipulation

Since the IAM source images already contain single words, there is no need to trim or crop them further. In addition, no noise removal techniques need to be applied since they already include plain backgrounds.

However, the data needs to be transformed. This can be done by normalizing the image which converts the pixels to have a value between 0 and 255. The pixel values can then be divided by 255 to scale the values to be in the range of [0,1], and then rounded to be either 0 or 1, which corresponds to black and white, respectively.

The process of normalization is critical to help the model converge faster during training by making the input values smaller, more consistent, and reducing skewness.

After normalization the images are fit into a canvas of size 64px x 129px. Empty space is filled with black pixels, and if an image is much smaller than the desired size, it is scaled up.

**Prepared Image**
![PreparedImage](https://github.com/yzzy2go/HandwritingOCR/assets/52092038/4b8563ac-77e2-4ec3-83c4-69114dcf21a3)


### Machine Learning

The first iteration of our model saw its loss and validation loss never converge to lower values. This meant that during the training stage, the model was unable to reach a stable state where loss was at a minimum and parameters weren’t changing significantly.

To improve upon Iteration 1, multiple parameters that are used in our CNN model were adjusted to improve the accuracy of our model. Specifically, hyperparameters, including the number of epochs, the number of layers in the model, and the optimizer type were experimented with to create later iterations.

The final model featured much smaller loss values, however, testing loss was higher than training loss which could be indicative of overfitting.

![FinalModelLoss](https://github.com/yzzy2go/HandwritingOCR/assets/52092038/169daa3b-60ac-4c83-b398-df76427d24cb)

The final model features 2 RNN layers, 3 CNN layers, and an Adam optimizer run over 10 epochs.

#### Parameter Tuning

To reach the final model, we experimented with different parameters to see how they would affect the end results.

The number of epochs was determined by evaluating the loss function with the objective to minimize the loss on a testing set. With too few epochs, it was possible that the model would unfit the data. This is because the model would not have learned the underlying data patterns enough. On the other hand, it was also important not to overfit the data with an overly large value for the epoch. Thus, as the number of epochs increased, we ensured that the testing set did not perform better than in the testing set through the monitoring of the loss value.

Similarly, the number of layers used in the model is another hyperparameter that was adjusted. Increasing the number of layers, which is also referred to as the depth of the model, can help increase the capacity of the model as well as make it more computationally efficient [1](https://machinelearningmastery.com/how-to-control-neural-network-model-capacity-with-nodes-and-layers/).

We also looked into adjusting the type of optimizer we used. The two optimizers used in our project were Stochastic Gradient Descent (SGD) and Adam (Adaptive Moment Estimation). In SGD, model parameters are updated after each loss computation on a training sample. Because of the increased number of updates, this optimizer converges in less time and requires less memory. However, SGD uses a fixed training rate and model parameters tend to have high variance [2](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6). In contrast, Adam updates parameters by computing a running average of the gradients of the first and second moments. In general, this optimizer will converge faster and use less resources [3](https://doi.org/10.3390/s20123344). We chose to change from SGD to Adam because research shows that Adam can often outperform other optimizers when used for handwriting recognition with CNNs [4](https://www.ibm.com/topics/overfitting).

### Accuracy Calculations

Initially, accuracy was determined by the percentage of exact matches of predicted to actual label. This was found to be the wrong metric, since looking at some outputs showed that the majority of the word was correctly predicted, with only a few individual characters differing.

A more accurate indicator of the model's performance was the edit distance, which is the number of insertions or removals required to reach the correct label.

This was calculated by comparing each word in a list of correct labels to a list of predicted labels from our model and returning the mean distance between the two.

The edit distance was 0.44 which indicates that, even though the "accuracy" was ~27%, on average, each incorrect word was only about 1 letter off.

