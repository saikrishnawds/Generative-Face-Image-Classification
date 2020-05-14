# Generative-Face-Image-Classification
This repository contains my implementation of Face image classification using Gaussian model, Mixture of Gaussian model, t-distribution and Factor Analyser.

Model 1 Learn single Gaussian model using training images and report your results
as stated above.
Model 2 Learn Mixture of Gaussian model using training images and report your
results as stated above. You can tune the number of components (e.g., based on cross validation
strategy).
Model 3  Learn t-distribution model using training images and report your results as
stated above.
Model 4  Learn factor analyzer using training images and report your results as
stated above. You can tune the dimensionality of the subspace.

Tasks: With your own face dataset created, you can train your models and test the performance.
For each model, report results as follows.

• Visualize the estimated mean(s) and covariance matrix for face and non-face
respectively; Use RGB images, but you are welcome to try on other things such as gray
images, gray images with histogram equalized, and HSI color space, etc.

• Evaluate the learned model on the testing images using 0.5 as threshold for the posterior.
Compute false positive rate (#negatives being classified as faces / #total negatives), and
false negative rate (#positives being classified as non-face / #total positives), and the
misclassification rate ((#false-positives + #false-negative)/#total testing images)

• Plot the ROC curve where x-axis represents false positive rate and y-axis true positive
rate (i.e, 1-false negative rate). To plot the curve, you change the threshold for posterior
from +∞ (use maximum posterior across testing images in practice, then all being
classified as non-faces) to −∞ (use minimum posterior across testing images in practice,
then all being classified as faces) with for example1000 steps. (ref:
https://en.wikipedia.org/wiki/Receiver_operating_characteristic )
