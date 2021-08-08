# Machine Learning Resources

This repository contains resources that I create from time to time.
I am still experimenting with the format of these courses, so any feedback will be highly appreciated.

Following type of resources are available

- Courses: These include two types of material
  - Lessons: This section includes 1-2 hour worth of relevant topics presented in the form of a jupyter notebook (each cell is a slide).
  - Practical: This section is about hands-on practical of a relevant topic applied in practice. There are two types of notebooks in this section:
    - `<name>-solution.ipynb`: This is a full solution to the practical. The notebook has been run to make sure there are no bugs.
    - `<name>.ipynb`: If you are interested in doing the practical yourself, this notebook is created from `<name>-solution.ipynb` after removing critical portions of the code necessary to grasp the relevant concepts.

- Tutorials: These are similar to `courses/practical` except that they focus on full implementation of a research paper. These include two types of notebooks:
  - `practical-solution.ipynb`: Similar to courses, this notebook is a full step-by-step guided tutorial of the research paper.
  - `practcal.ipynb`: This notebook removes critical portion of the code for students to fill in the solution.


- Hackathons: These are applied AI problems with two types of notebooks:
-
  - `<name>.ipynb`: If the hackathon is based upon a published research paper, this notebook reproduces the results of that paper. It enables me to think through possible avenues that can be left open to design a hackathon.
  - `<name>_hackathon.ipynb`: This notebook presents the hackathon in the form of questions to guide solution design.


## Courses

1. [Noise reduction](courses/noise-reduction):
   1. Lessons:
      1. Sources of noise
      2. Impact of noise
      3. Examples of noise
      4. Denoising techniques
         1. Rolling Window
         2. Convolution
         3. Digital Filters
         4. Machine Learning Filters
         5. Denoising Autoencoders (DAE)
   1. Practical: Build and train a DAE on Fashion MNIST dataset.


2. [Deep Autoencoders and Variational Autoencoders](courses/deep-autoencoders)
   1. Lessons:
       1. Autoencoders: Motivation & History
       2. Autoencoders: Loss function
       3. Undercomplete Autoencoder
       4. Overcomplete Autoencoders
       4. Stacked / Deep Autoencoders
       5. Variational Autoencoders
       6. Applications
   2. Practical: Build and train autoencoders with different types of regularizations (e.g., ) to enable data compression, data generation and data interpolation. Example dataset is MNIST Digit dataset


3. [Advanced CNN architectures (2012-2018)](courses/mna/)
   1. Lessons:
      1. History
      2. Network in Network
      3. InceptionNet
      4. ResNet
      5. Pre-activations: Improved ResNet
      6. DenseNet
      7. WaveNet
      8. Depthwise Separable Convolutions
      9. Squeeze-and-Excite: ResNet
      10. Neural Architecture Search  
   2. Practical: Build and train InceptionNet and DenseNet on CIFAR-10 dataset  


## Tutorials

1. [Attention is all you need](tutorial/attention_is_all_you_need): It is a step-by-step guide to build a machine translation model as proposed by [Vaswani et al. Attention is all you need](https://arxiv.org/abs/1706.03762). Build and train a transformer model on [Machine Translation Task using the Flickr30k dataset](http://www.statmt.org/wmt16/multimodal-task.html#task1). The trained model is further used to infer the likely translations using (a) greedy decoding and (b) beam search inference procedures.


## Hackathons

1. Prediction of COVID from symptoms: Build a practical classifier to detect COVID from reported symptoms. It is based on the work by [Zoabi et al. Machine learning-based prediction of COVID-19 diagnosis based on symptoms. npj Digit. Med. 4, 3 (2021).](https://www.nature.com/articles/s41746-020-00372-6)
