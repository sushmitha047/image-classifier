import argparse
import torch
from PIL import Image
import json
from torchvision import models
import numpy as np
import sys
from contextlib import redirect_stdout
from time import time
import os

from torch.serialization import add_safe_globals
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear
from torch.nn.modules.activation import ReLU, LogSoftmax
from  torch.nn.modules.dropout import Dropout
from torch.nn.modules.loss import NLLLoss

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, help='CNN Model Architecture')
    parser.add_argument('--path', type=str, help='Path to the test picture')
    parser.add_argument('--categories', type=str, help='File with the class names')
    parser.add_argument('--topk', type=int, help='Amount of top classes')
    parser.add_argument('--gpu', type=bool, help='GPU usage')

    return parser.parse_args()


def input_validation(args):
    if args.arch:
        if args.arch == 'vgg' or args.arch == 'densenet':
            checkpoint_path = './model_checkpoints/checkpoint_' + args.arch + ".pth"
            if args.arch == 'vgg':
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            elif args.arch == 'densenet':
                model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            print('Invalid model choice. Using checkpoint of the default model: densenet')
            checkpoint_path = './model_checkpoints/checkpoint_densenet.pth'
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    else:
        print('Using checkpoint of the default model: densenet')
        checkpoint_path = './model_checkpoints/checkpoint_densenet.pth'
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    if args.path:
        path = args.path
    else:
        path = "./flower_data/test/15/image_06351.jpg"
        print("Default picture path: {}".format(path))

    if args.categories:
        categories = args.categories
    else:
        categories = "cat_to_name.json"
        print("Default categories path: {}".format(categories))

    if args.topk:
        topk = args.topk
    else:
        topk = 5
        print("Default amount of top classes: {}".format(topk))

        

    if args.gpu:
        if args.gpu:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("Using default cpu")

    return checkpoint_path, model, path, categories, topk, device

# function to load large model checkpoint file split into multiple files
def load_split_checkpoint(base_path):
    checkpoint = {}
    chunk_id = 0
    while True:
        chunk_path = f"{base_path}.part{chunk_id}"
        if not os.path.exists(chunk_path):
            break
        with open(chunk_path, 'rb') as f:
            chunk_data = torch.load(f, weights_only=True)
            checkpoint.update(chunk_data)
        chunk_id += 1

    return checkpoint



def get_checkpoint(path, model):
    # Add Sequential to safe globals before loading checkpoint
    add_safe_globals([Sequential, Linear, set, ReLU, Dropout, LogSoftmax, NLLLoss])

    # Load split checkpoint
    if os.path.exists(f"{path}.part0"):
        checkpoint = load_split_checkpoint(path)
    else:
        checkpoint = torch.load(path, weights_only=True)


    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_index_connection']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(imagepath):
    im = Image.open(imagepath)

    width, height = im.size

    shortest_side = 256
    if width > height:
        ratio = width / height
        height = shortest_side
        width = shortest_side * ratio
    elif height > width:
        ratio = height / width
        width = shortest_side
        height = shortest_side * ratio
    else:
        height = shortest_side
        width = shortest_side

    im.thumbnail((width, height), Image.Resampling.LANCZOS)

    x0 = (width - 224) / 2
    y0 = (height - 224) / 2
    x1 = x0 + 224
    y1 = y0 + 224

    im = im.crop((x0, y0, x1, y1))

    image_to_np_array = np.array(im) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (image_to_np_array - mean) / std

    final_image = im.transpose((2, 0, 1))

    return final_image


def main():
    # Redirect stdout to capture outputs
    with open('predict_results.txt', 'w') as f:
        with redirect_stdout(f):
            start_time = time()

            in_arg = get_input_args()
            print(f"CNN Architecture: {in_arg.arch}\n")
            print(f"Image path : {in_arg.path}\n")
            print(f"Flower Categories: {in_arg.categories}\n")
            print(f"Top K predictions: {in_arg.topk}\n")
            print(f"Device == GPU: {in_arg.gpu}\n")
            print()
            # print(in_arg.arch, in_arg.path, in_arg.categories, in_arg.topk, in_arg.gpu)
            checkpoint_path, model, path, categories, topk, device = input_validation(in_arg)

            with open(categories, 'r') as f:
                cat_to_name = json.load(f)

            model = get_checkpoint(checkpoint_path, model)

            model.eval()
            model.to(device)

            array_image = process_image(path)

            tensor = torch.from_numpy(array_image).float().unsqueeze(0)
            tensor = tensor.to(device)

            log_predictions = model.forward(tensor)
            predictions = torch.exp(log_predictions)
            top_probs, top_classes = predictions.topk(topk)

            top_probs = np.array(top_probs.detach().cpu(), dtype=np.float32, copy=None)[0]
            top_classes = np.array(top_classes.detach().cpu(), dtype=np.float32, copy=None)[0]

            idx_to_class = {value: key for key, value in model.class_to_idx.items()}
            top_classes = [idx_to_class[label] for label in top_classes]

            top_classes_name = [cat_to_name[class_num] for class_num in top_classes]


            end_time = time()
            tot_time = end_time - start_time

            print("Top probabilities: {} ".format(top_probs))
            print("Top classes: {} ".format(top_classes))
            print("Top classes' names: {} ".format(top_classes_name))

            print("\n** Total Elapsed Runtime:",
                  str(int((tot_time / 3600))) + ":" + str(int((tot_time % 3600) / 60)) + ":"
                  + str(int((tot_time % 3600) % 60)))

if __name__ == "__main__": main()