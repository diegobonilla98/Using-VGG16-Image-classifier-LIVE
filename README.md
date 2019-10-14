# Using-VGG16-Image-classifier-LIVE
My easy approach to using the pretrained network.

Very easy. Load the pretrained model VGG16 (various layers of Conv2D and MaxPooling2D followed by a basic classifier neural network) and after some image preprocessing (crop the image, transform it into a np.array, reshape it, ...) just to match the wanted input of the model and then proceed with the predictions.
