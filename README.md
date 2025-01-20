# Deep Learning Image Classifier for Flower Species

## Project Overview
This project is part of the AWS AI/ML Scholarship Program Nanodegree - AI Programming with Python. It implements a deep learning model that can recognize and classify 102 different flower species using transfer learning with PyTorch. The project demonstrates core machine learning techniques including data preprocessing, neural network training, and inference.

## Project Structure
 - `Image_Classifier_Project.ipynb`: Jupyter notebook containing the development process
 - `train.py`: Script for training the neural network and saving checkpoints
 - `predict.py`: Script for using the trained network to make predictions
 - `cat_to_name.json`: Mapping of categories to flower names

 ## Features
 - Transfer learning using pre-trained networks (VGG16 and DenseNet121)
 - Data augmentation for improved model generalization
 - Command-line interface for both training and prediction
 - GPU acceleration support (CUDA and MPS)
 - Checkpoint saving and loading for model persistence

## How to Run
1. Clone the repository: 
```
    git clone https://github.com/yourusername/image-classifier.git
```
```
    cd image-classifier
```

2. Install the required dependencies:
```
    pip install -r requirements.txt
```

3. Training: Train a new network on a dataset
```
    python train.py --arch [architecture] --rate [learning rate] --hiddenUnits [hidden layer size] --epochs [epochs] --gpu [True/False]
```
Options:
 - `--arch`: Choose model architecture (densenet or vgg)
 - `--rate`: Set learning rate
 - `--hiddenUnits`: Specify hidden layer sizes
 - `--epochs`: Number of training epochs
 - `--gpu`: Enable GPU training if available (True or False)

4. Prediction: Use the trained network to predict flower species
```
   python predict.py --arch [architecture] --path 'path/to/image.jpg' --categories 'cat_to_name.json' --topk [No. of top predictions] --gpu [True/False]
```
Options:
 - `--arch`: Model architecture to use (densenet or vgg)
 - `--path`: Path to input image
 - `--categories`: JSON file mapping categories to flower names
 - `--topk`: Number of top predictions to show
 - `--gpu`: Enable GPU inference (True or False)

## Model Architecture
 - The classifier uses transfer learning with two options:
 - DenseNet121: Pre-trained on ImageNet, modified with custom classifier
 - VGG16: Pre-trained on ImageNet, modified with custom classifier
 - Both architectures are fine-tuned for the specific task of flower classification.

## Dataset
 - The model is trained on a dataset of [102 flower categories](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). The data is split into:
 - Training set: Used for model training
 - Validation set: Used for hyperparameter tuning
 - Test set: Used for final model evaluation

## Project Results
Three pre-trained networks were evaluated for the flower classification task using the same hyperparameters (10 epochs, learning rate 0.003) except for hidden units:

| Model      | Hidden Units | Training Time | Final Validation Accuracy |
|------------|--------------|----------------|---------------------------|
| DenseNet121| 512, 256     | ~9m            | 87.9%                     |
| VGG16      | 4096         | ~11m           | 78.6%                     |
| ResNet50   | 512, 256     | ~10m           | 82.8%                     |

Key Findings:
 - DenseNet121 achieved the best performance with 87.9% validation accuracy
 - All models showed good convergence with decreasing loss trends
 - DenseNet121 demonstrated better generalization with lowest validation loss

Prediction Examples:
 - DenseNet121 correctly identified a globe-flower with 95.9% confidence
 - VGG16 identified a globe thistle with 99.9% confidence
 - ResNet50 identified a poinsettia with 99.95% confidence
 - This analysis shows that DenseNet121 provides the best balance of accuracy, training time, and model complexity for this flower classification task.

## Acknowledgments
 - AWS AI/ML Scholarship Program
 - Udacity

## License
This project is licensed under the MIT License - see the LICENSE file for details.