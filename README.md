# Project Title
# TextGAN: Unsuperivsed Text Segmentation
Text segmentation is a difficult problem because of the potentially vast variation in text and scene landscape. Moreover, systems that learn to perform text segmentation usually need non-trivial annotation efforts. This repositry conaints the implementation of unsupervised method to segment text at the pixel-level from scene images. The model we propose, which relies on generative adversarial neural networks, segments text intelligently; and does not therefore need to associate the scene image that contains the text to the ground-truth of the text. The main advantage is thus skipping the need to obtain the pixel-level annotation dataset, which is normally required in training powerful text segmentation models. The code is basesd on PyTorch 1.0.0 and might also work with >=0.4 versions.

Trained models can be found in [text_segmentation256-Jun-2](https://github.com/morawi/TextGAN/tree/master/text_segmentation256-Jun-2). Each model has been built using only 9 residual blocks.


### Prerequisites

Numpy; 
PIL

### Installing

Download or clone the repositry, and off you go


## Running the tests
To train the model, use [CycleGAN_text.py](https://github.com/morawi/TextGAN/blob/master/cyclegan_text.py)
To test the model, use [test_GAN_AB.py](https://github.com/morawi/TextGAN/blob/master/test_GAN_AB.py)


## Author

* **Mohammed Al-Rawi** - 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
* Inspiration? The magic of GANs
* [PyTorch] (http://pytorch.org)

## Text Segmentation Samples
![Samples](https://github.com/morawi/TextGAN/blob/master/generated_samples/0.png)
