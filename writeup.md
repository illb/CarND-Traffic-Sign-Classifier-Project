#**Traffic Sign Recognition** 

##Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-data/visualization.png "Visualization"
[image2]: ./writeup-data/rotations.png "rotations"
[image3]: ./writeup-data/grayscale.png "grayscale"
[image4]: ./writeup-data/01-08.png "Traffic Sign 1"
[image5]: ./writeup-data/02-14.png "Traffic Sign 2"
[image6]: ./writeup-data/03-18.png "Traffic Sign 3"
[image7]: ./writeup-data/04-12.png "Traffic Sign 4"
[image8]: ./writeup-data/05-33.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/illb/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration3
![alt text][image2]
####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the c for any color
de, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.


* step 2) append the gray scale image
  * to detect for any color
![alt text][image3]

* step 3) normalize images
  * move to center, and shrink to a absolute value of 1.0
  * to training faster and to avoid local optima

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

* I used given validation and testing data
* I added rotated traing images -20, -10, 10, 20 degrees
  * to detect signs at various angles
  * my training set was 5 times

![alt text][image2]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x4 RGB & gray image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6				 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 5x5x16					|
| Flatten				| outputs 400        							|
| dropout				| keep probability 0.7							|
| Fully connected		| outputs 120        							|
| RELU					|												|
| dropout				| keep probability 0.7							|
| Fully connected		| outputs 84        							|
| RELU					|												|
| dropout				| keep probability 0.7							|
| Fully connected		| outputs 43        							|
| Softmax				|        										| 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

* batch size : 128
* epochs : 40
* optimizer : RMSPropOptimizer
  * learning rate : 0.001
  * decay : 0.9
* traing operation : minimize cross entropy of softmax probabilites with one hot encoded class ids
* training dropout keep probability : 0.7

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * From lenet, the input channel is increased to 3, allowing color input.
* What were some problems with the initial architecture?
  * when colors change cannot detect
  * rotated images cannot detect
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * add grayscale channel
  * add rotated images (-20, -10, 10, 20 degrees)
  * normalize images
  * add dropout (0.7) to avoid overfit
  * wider layers (2 times)
  * use xavier initializer to fater training
* Which parameters were tuned? How were they adjusted and why?
  * increase epoch to 50
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
  * modified Lenet
* Why did you believe it would be relevant to the traffic sign application?
  * Lenet is simple and easy to modofy
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.958 
* test set accuracy of 0.944
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (120km/h) | Speed limit (120km/h)    						| 
| Stop     				| Stop											|
| General caution		| General caution								|
| Priority road     	| Priority road					 				|
| Turn right ahead		| Turn right ahead								|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

* Speed limit (120km/h)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (120km/h)							| 
| .00     				| 												|
| .00					| 												|

* Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop											| 
| .00     				| 												|
| .00					| 												|

* General caution

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General caution								| 
| .00     				| 												|
| .00					| 												|

* Priority road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road									| 
| .00     				| 												|
| .00					| 												|

* Turn right ahead

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn right ahead								| 
| .00     				| 												|
| .00					| 												|

