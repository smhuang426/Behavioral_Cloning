# **Behavioral Cloning** 

Result Demo
[![Watch the demo video]](https://youtu.be/lcO3vkUIxJo)
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/crop.png "Crop"
[image2]: ./examples/flip.png "Flip"
[image3]: ./examples/resize.png "Resize"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2 * 2 ~ 8 * 8 filter sizes and depths between 8 and 128 (model.py lines 121-153) 

The model includes RELU layers to introduce nonlinearity (code line 123), and the data is normalized before the model using Opencv to crop it for removing sky and resize input to 64 * 64 * 3 (code line 67). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 131). I add it on fully connection layer. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 155).

#### 4. Appropriate training data

I using the training data provided by Udacity course in project resources section, because I'm not a good racing car game player, so there is a big quetion, I have no enough training data to train my model, I need to augment it.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was reduce computation first....
Because I have no GPU and applied AWS service fail.
If I use 160 * 360 * 3 image size with CPU, I will need to spend 2.5 hours per epoch.

My first step was resizing input image to 64 * 64 * 3, and it work, just spend 40 mins per epoch. 

And I use a convolution neural network model similar to the lenet but more layer in my CNN without pooling layer, I thought this model might be appropriate because more layer maybe had good result. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I make my model smaller, and use samller scale on fully connection layer. Besides, I add dropout on fully connection layer too. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when turn right, because left turn case is more more then right turn case. To improve the driving behavior I do data aguement,  I will introduce in section 3.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 64x64x3 RGB image   							| 
| Convolution 8x8     	| 4x4 stride, SAME padding, outputs 16x16x8 	|
| RELU					|												|
| Convolution 8x8	    | 4x4 stride, SAME padding, outputs 4x4x16 |
| RELU					|												|
| Convolution 4x4	    | 2x2 stride, SAME padding, outputs 2x2x32 |
| RELU					|												|
| Convolution 2x2	    | 1x1 stride, SAME padding, outputs 2x2x64 |
| RELU					|												|
| Droupout					|												|
| Fully connected		| Input = 256. Output = 64 |
| Droupout					|												|
| Fully connected		| Input = 64. Output = 32 |
| Fully connected		| Input = 32. Output = 1 |

#### 3. Creation of the Training Set & Training Process

I use the images provided by course.

I then crop the image to remove sky part

![alt text][image1]


Then I resize training set to 64 * 64 * 3 so that I can use small scale CNN to achieve low computation and good result.
![alt text][image3]

But I found that, there is less right turn data.
To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image2]

After the augment the data set, I had around 9000 of data points. I only flip those images if steering number is bigger than 0.05.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1. I used an adam optimizer so that manually training the learning rate wasn't necessary.
