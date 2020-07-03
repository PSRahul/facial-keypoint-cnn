# Facial Keypoint Detector 

This deep learning model based on convolutional neural networks detect the facial keypoints of left eye , right eye and mouth.

The model has 7 Convolutional layers and 3 Linear layers.
Implemented with pytorch.
Uses Tensorboard for learning visualisation.

Uses MSELoss and Adagrad Optimiser.

Trained Using dataset from Kaggle
https://www.kaggle.com/c/facial-keypoints-detection/data
"The data set for this competition was graciously provided by Dr. Yoshua Bengio of the University of Montreal."

Sample images provide model predictions vs. the true values.

Future Possible Work
1.Augmented train dataset with manual labelling of keypoints
2.Using advanced layers such as skip and residual nets.
