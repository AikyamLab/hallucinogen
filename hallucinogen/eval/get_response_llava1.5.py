from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import tqdm
import os
import json
from llava.model import *
import torch
from peft import PeftModel
from llava.model.builder import load_pretrained_model_rlhf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.manifold import TSNE
import argparse

model_path = "liuhaotian/llava-v1.5-7b"

COUNTERFACTUAL_PROMPTS = [
    "How would the scene change if the <obj> were invisible? Provide a concise answer",
    "What might replace the <obj> if it vanished from the image? Provide a concise answer",
]

IDENTIFICATION_PROMPTS = [
    'Is the <obj> present in the image?',
    'Determine whether a <obj> is visible in the image.',
]


LOCALIZATION_PROMPTS = [
    'Describe where is <obj> in the image.',
    'Examine the presence and location of <obj> in the image.',
]

VISUAL_CONTEXT_PROMPTS = [
    'Interpret the object <obj> with respect to its context within the image. Provide a concise answer',
    'Analyze the neighboring elements of <obj> in the image. Provide a concise answer',
]
def generate_output(image_file, prompt, label):
    #prompt = "What are the things I should be cautious about when I visit here?"
    #image_file = "https://llava-vl.github.io/static/images/view.jpg"
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "label": label,
        "max_new_tokens": 256
    })()

    return eval_model(args)


def main_file_predict(adv_split, split_num):
    image_dir = '/fs/nexus-scratch/aseth125/val2014'
    print(adv_split)
    image_dir = '/fs/nexus-scratch/aseth125/val2014'
    content_output = []
    content = []
    with open(adv_split) as file:
        for lin in file:
            content.append(json.loads(lin))
        
        for data in tqdm.tqdm(content):
            image_path = os.path.join(image_dir, data["image"])
            obj = ' '.join(data["text"].split(' ')[3:-3])
            
            for i, prompt in enumerate(COUNTERFACTUAL_PROMPTS):
                data[f'query_{i+1}'] = prompt.replace('<obj>', obj)
                data[f'output_{i+1}'] = generate_output(image_path, data[f'query_{i+1}'], None)[0]
            
            content_output.append(data)

    with open(f'count_response_{split_num}_'+adv_split.split('/')[-1],'w') as file:
        for data in content_output:
            json.dump(data, file)
            file.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Split a list into equal parts.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        required=True, 
        help="Path to the file containing the list."
    )
    parser.add_argument(
        "--split_num", 
        type=int, 
        required=True, 
        help="Number of sublists to split into."
    )

    args = parser.parse_args()
    main_file_predict(args.file_path, args.split_num)
if __name__ == '__main__':
    main()

