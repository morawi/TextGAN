# TextGAN: Unsuperivsed Text Segmentation
Text segmentation is a difficult problem because of the potentially vast variation in text and scene landscape. Moreover, systems that learn to perform text segmentation usually need non-trivial annotation efforts. This repositry conaints the implementation of unsupervised method to segment text at the pixel-level from scene images. The model we propose, which relies on generative adversarial neural networks, segments text intelligently; and does not therefore need to associate the scene image that contains the text to the ground-truth of the text. The main advantage is thus skipping the need to obtain the pixel-level annotation dataset, which is normally required in training powerful text segmentation models. The code is basesd on PyTorch 1.0.0 and might also work with >=0.4 versions.

Trained models can be found in [text_segmentation256-Jun-2](https://github.com/morawi/TextGAN/tree/master/text_segmentation256-Jun-2). Each model has been built using only 9 residual blocks.


### Prerequisites

Numpy; 
PIL

### Installing

Download or clone the repositry, and off you go

## Datasets
Place the training and testing samples in two separate folders calle tarin and test, respectively. Each folder should have the scene-text images in a folder calle A and the pixel-wise level annotations in another folder called B. The testing folder should have paired images to verify the performance via F1, but the training folder can have unpaired images. This is a simple and straightforward strategy, you just need to copy your images into these folders. The default name of the folder containing these train and test folders is called 'text_segmentation256', but can be changed by the user accordingly. The folder 'text_segmentation256' is placed outside the implementation, so make sure to correct the path according to your folder's path. 

## Running the tests
To train a model, use [CycleGAN_text.py](https://github.com/morawi/TextGAN/blob/master/cyclegan_text.py);
To test the model, use [test_GAN_AB.py](https://github.com/morawi/TextGAN/blob/master/test_GAN_AB.py)


## Author

* **Mohammed Al-Rawi** - 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
* Inspiration? The magic of GANs
* [PyTorch](http://pytorch.org)

## Text Segmentation Samples via CycleGAN 
input,            GAN_AB(input),             -ve(input),              GAN_AB(-ve(input))
![Samples](https://github.com/morawi/TextGAN/blob/master/generated_samples/0.png)
![Samples](https://github.com/morawi/TextGAN/blob/master/generated_samples/1.png)
![Samples](https://github.com/morawi/TextGAN/blob/master/generated_samples/2.png)
![Samples](https://github.com/morawi/TextGAN/blob/master/generated_samples/3.png)
![Samples](https://github.com/morawi/TextGAN/blob/master/generated_samples/4.png)
![Samples](https://github.com/morawi/TextGAN/blob/master/generated_samples/5.png)
![Samples](https://github.com/morawi/TextGAN/blob/master/generated_samples/6.png)



