# emoji
In the machine learning world, there is a famous data set known by practically everybody who has worked with classification models. This data set is the MNIST database of handwritten digits (Enlaces a un sitio externo.). It contains thousands of images of digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) written by humans. This project is inspired by the work conducted by many researchers to collect and analyze these types of databases.

The goal of this project is to build a small data set of hand-drawn emojis and test various machine learning approaches to recognize these images. You will test three different learning strategies in machine learning (supervised, unsupervised, and reinforcement).  Additionally, you will work with different classification models (SVM, neural networks, KNN ) and clustering methods (K-means and dendrograms).

The project is organized as follows:

## Data collection
To create a database of hand-drawn emojis, everybody in the class must contribute with 100 images. The images must be square (for example 1200 x 1200), with a solid background (white or something close to white). The emoji must be centered, covering almost all the image, drawn with black lines. You can use your mobile to draw the emojis with your finger or a digital pen. 

Five types of emojis must be included in the dataset: happy faces, sad faces, angry faces, surprised faces, and poos. You must add at least 20 images of each class of emoji. The following images show the emojis and some examples of images.

<div>
<a href="url"><img src="https://i.ibb.co/CnTp0CZ/Happy.png" align="center" height="100" width="100"></a>
<a href="url"><img src="https://i.ibb.co/dJsRvC1/Happy-example.jpg" align="center" height="100" width="100"></a>
</div>


Add your images to this [shared folder](https://tecmx-my.sharepoint.com/:f:/g/personal/omendoza83_tec_mx/Emill4lWQIlAsYMgqUKOxS4BrR0W05UxoWaZqGgV6NgLZA?e=rPHfqF) in the respective subfolder of each emoji. Your images must be named starting by the initials of your given name followed by the number of the image. For example, my first happy face image is named as omm_1.jpg.

The deadline to add your images to the shared folder is on Monday 18 at 11:59 pm. 

### Hint:

Install [Adobe Sketch](https://play.google.com/store/apps/details?id=com.adobe.creativeapps.sketch&hl=en) in your mobile, and use this app to create your images. Use square images and a black pencil to draw your emojis. Finally, press the share button of this app to publish your images, and download them from [Behance](https://www.behance.net/).

## Data normalization and pre-processing
Usually, the original images of the MNIST database are re-mixed, normalized, and pre-processed before fitting a model. This is an important operation because machine learning models won't work with this kind of samples without transforming the original data.

In this stage of the project, download all the available images in the [Emojis database](https://tecmx-my.sharepoint.com/:f:/g/personal/omendoza83_tec_mx/Emill4lWQIlAsYMgqUKOxS4BrR0W05UxoWaZqGgV6NgLZA?e=rPHfqF), (Enlaces a un sitio externo.) and pre-process and normalize the images. You can try any normalization approach, but the recommended steps are the following for each image:

Binarize the image (white background, black lines).
Identify the bounding rect of the emoji.
Rescale the image inside the bounding rect to a size of 32 x 32.
Save the new small image.
Hint: You can use PCA to compute new features for the Emojis dataset. CNNs does not need these pre-process step, but the performance of SVMs and other classifiers may be improved using this technique.

## Supervised learning
Train at least two types of classifiers (RB-SVM and KNN for example), and determine which classes can be identified by the models. Remember to use cross-validation to evaluate the performance of the classifiers. Additionally, train a convolutional neural network with this data, and compare the results obtained with the different models.

## Unsupervised learning
Use K-Means and dendrograms to identify possible groups of similar images in the data set. Try different numbers of groups in the K-means method.

Are these groups organized according to the types of emojis?

## Conclusions
Write some conclusions about this work. Let me know what you have learned, and what you would like to improve. Conclusions are individual. Each team member must write at least 400 words.
