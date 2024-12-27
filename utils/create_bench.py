import requests
import torch
from PIL import Image
import os
import json
import tqdm
import argparse


COUNTERFACTUAL_PROMPTS = [
    "How would the scene change if the <obj> were invisible?",
    "What might replace the <obj> if it vanished from the image?",
    "What is the purpose of the <obj> in this scene, and can the image work without it?",
    "If <obj> were moved elsewhere, what would happen to the story of the image?",
    "What would happen if we removed the <obj> from the image?",
]

IDENTIFICATION_PROMPTS = [
    'Is the <obj> present in the image?',
    'Determine whether a <obj> is visible in the image.',
    'Assess if a <obj> can be seen in the image.',
    'Find out if a <obj> appears in the photo.',
    'Verify if there is a <obj> in the image.',
]


LOCALIZATION_PROMPTS = [
    'Describe where is <obj> in the image.',
    'Examine the presence and location of <obj> in the image.',
    'Locate and describe the position of <obj> in the image.',
    'Determine and specify where <obj> is in the photo.',
    'Explain where <obj> is found in the image.',
]

VISUAL_CONTEXT_PROMPTS = [
    'Interpret the object <obj> with respect to its context within the image.',
    'Analyze the neighboring elements of <obj> in the image.',
    'Describe the context and surroundings of <obj> in the picture.',
    'Explain the context in which <obj> is placed within the image.',
    'Outline the context and nearby items around <obj> in the photo.',
]

IDENTIFICATION_PROMPTS_XRAY = [
    'Given this X-ray, identify if the person has <obj>.',
    'Based on this X-ray, determine whether the person has <obj>.',
    'Analyze this X-ray to identify if <obj> is present in the person.',
    'Examine this X-ray and conclude if the person has <obj>.',
    'Review this X-ray to assess whether the person shows signs of <obj>.'
]


LOCALIZATION_PROMPTS_XRAY = [
    'Examine the X-ray and identify the region associated with detecting <obj>.',
    'Analyze the X-ray and determine which region is linked to <obj>.',
    'Inspect the X-ray and specify the area corresponding to <obj>.',
    'Evaluate the X-ray to locate regions indicative of <obj>.',
    'Review the X-ray and pinpoint the region associated with identifying <obj>.',
]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a list into equal parts.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        default='/fs/nexus-projects/brain_project/ashish/img_exp/hallucinogen/med_hallucinogen/data/med_hallucinogen_data.json',
        help="Path to the file containing the list."
    )

    args = parser.parse_args()
    split_path = args.file_path
    image_dir = '/fs/nexus-scratch/aseth125/val2014'
    content_output = []
    content = []
    with open(split_path) as file:
        for lin in file:
            content.append(json.loads(lin))
        
        for data in tqdm.tqdm(content):
            image_path = os.path.join(image_dir, data["image"])
            obj = data['diseases']
            
            for i, prompt in enumerate(LOCALIZATION_PROMPTS_XRAY):
                data[f'query_{i+1}'] = prompt.replace('<obj>', obj)
                #data[f'output_{i+1}'] = get_response(image_path, data[f'query_{i+1}'])
                print(data.keys())
            content_output.append(data)

    with open(f'loc/loc_'+split_path.split('/')[-1],'w') as file:
        for data in content_output:
            json.dump(data, file)
            file.write('\n')



