# Machine Learning Resources

This repository contains resources that I create from time to time.
I am still experimenting with the format of these resources, so any feedback will be highly appreciated.

Following type of resources are available:

- **Courses**: These include two types of material
  - *Lessons*: This section includes 1-2 hour worth of relevant topics presented in the form of a jupyter notebook (each cell is a slide).
  - *Practical*: This section is about hands-on practical of a relevant topic applied in practice. There are two types of notebooks in this section:
    - `<name>-solution.ipynb`: This is a full solution to the practical. The notebook has been run to make sure there are no bugs.
    - `<name>.ipynb`: If you are interested in doing the practical yourself, this notebook is created from `<name>-solution.ipynb` after removing critical portions of the code necessary to grasp the relevant concepts.

- **Tutorials**: These are similar to `courses/practical` except that they focus on full implementation of a research paper. These include two types of notebooks:
  - `practical-solution.ipynb`: Similar to courses, this notebook is a full step-by-step guided tutorial of the research paper.
  - `practcal.ipynb`: This notebook removes critical portion of the code for students to fill in the solution.


- **Hackathons**: These are applied AI problems with two types of notebooks:
  - `<name>.ipynb`: If the hackathon is based upon a published research paper, this notebook reproduces the results of that paper. It enables me to think through possible avenues that can be left open to design a hackathon.
  - `<name>_hackathon.ipynb`: This notebook presents the hackathon in the form of questions to guide solution design.


## Courses

1. [Noise reduction](courses/noise-reduction):
   1. [Lessons](courses/noise-reduction/lessons/noise-reduction-interactive-slides.ipynb):
      1. Sources of noise
      2. Impact of noise
      3. Examples of noise
      4. Denoising techniques
         1. Rolling Window
         2. Convolution
         3. Digital Filters
         4. Machine Learning Filters
         5. Denoising Autoencoders (DAE)
   2. [Practical](courses/noise-reduction/practical/noise-reduction-pratcical-solution.ipynb) (Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)): Build and train a DAE on [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).


2. [Deep Autoencoders and Variational Autoencoders](courses/deep-autoencoders)
   1. [Lessons](courses/deep-autoencoders/lessons/deep-autoencoders-slides-interactive.ipynb):
       1. Autoencoders: Motivation & History
       2. Autoencoders: Loss function
       3. Undercomplete Autoencoder
       4. Overcomplete Autoencoders
       4. Stacked / Deep Autoencoders
       5. Variational Autoencoders
       6. Applications
   2. [Practical](courses/deep-autoencoders/practical/deep-autoencoders-practical-solution-full.ipynb) (Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)): Build and train autoencoders on [MNIST Digit dataset](http://yann.lecun.com/exdb/mnist/) with different types of regularizations (e.g., ) to enable data compression, data generation and data interpolation.


3. [Advanced CNN architectures (2012-2018)](courses/mna/)
   1. [Lessons](courses/mna/lessons/mna-interactive.ipynb):
      1. History
      2. Network in Network (2014)
      3. InceptionNet (2014)
      4. ResNet (2015)
      5. Pre-activations: Improved ResNet (2016)
      6. DenseNet (2016)
      7. WaveNet (2016)
      8. Depthwise Separable Convolutions (2017)
      9. Squeeze-and-Excite: ResNet (2017)
      10. Neural Architecture Search (2018 - Ongoing)
   2. [Practical](courses/mna/practical/mna-practical-solution.ipynb) (Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)): Build and train InceptionNet and DenseNet on [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)


## Tutorials

1. [Attention is all you need](tutorials/attention_is_all_you_need/practical-solution.ipynb) (Dataset: [Flickr30k](http://www.statmt.org/wmt16/multimodal-task.html#task1)): It is a step-by-step guide to build a machine translation model as proposed by [Vaswani et al. Attention is all you need](https://arxiv.org/abs/1706.03762). Build and train a transformer model on [Machine Translation Task using the Flickr30k dataset](http://www.statmt.org/wmt16/multimodal-task.html#task1). The trained model is further used to infer the likely translations using (a) greedy decoding and (b) beam search inference procedures.


## Hackathons

1. [Prediction of COVID from symptoms](hackathons/COVID_diagnosis/COVID_diagnosis.ipynb) (Dataset: [Covid cases](https://github.com/nshomron/covidpred)): Build a practical classifier to detect COVID from reported symptoms. It is based on the work by [Zoabi et al. Machine learning-based prediction of COVID-19 diagnosis based on symptoms. npj Digit. Med. 4, 3 (2021).](https://www.nature.com/articles/s41746-020-00372-6)
