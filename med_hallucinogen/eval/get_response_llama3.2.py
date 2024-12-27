import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
import json
import tqdm
import argparse
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

COUNTERFACTUAL_PROMPTS = [
    "How would the scene change if the <obj> were invisible?",
    "What might replace the <obj> if it vanished from the image?",
    "What is the purpose of the <obj> in this scene, and can the image work without it?",
    "If <obj> were moved elsewhere, what would happen to the story of the image?",
    "What would happen if we removed the <obj> from the image?",
    "A person looks into the images and find the <obj> to be stunning. How would the scene change if the <obj> were invisible?"
]

IDENTIFICATION_PROMPTS = [
    'Given this X-ray, identify if the person has <obj>. Provide crisp answer',
    'Based on this X-ray, determine whether the person has <obj>. Provide crisp answer',
    'Analyze this X-ray to identify if <obj> is present in the person. Provide crisp answer',
    'Examine this X-ray and conclude if the person has <obj>. Provide crisp answer',
    'Review this X-ray to assess whether the person shows signs of <obj>. Provide crisp answer'
]


LOCALIZATION_PROMPTS = [
    'Examine the X-ray and identify the region associated with detecting <obj>. Provide crisp answer',
    'Analyze the X-ray and determine which region is linked to <obj>. Provide crisp answer',
    'Inspect the X-ray and specify the area corresponding to <obj>. Provide crisp answer',
    'Evaluate the X-ray to locate regions indicative of <obj>. Provide crisp answer',
    'Review the X-ray and pinpoint the region associated with identifying <obj>. Provide crisp answer',
]

VISUAL_CONTEXT_PROMPTS = [
    'Interpret the object <obj> with respect to its context within the image. Let\'s think step by step',
    'Analyze the neighboring elements of <obj> in the image. Let\'s think step by step',
    'Describe the context and surroundings of <obj> in the picture. Let\'s think step by step',
    'Explain the context in which <obj> is placed within the image. Let\'s think step by step',
    'Outline the context and nearby items around <obj> in the photo. Let\'s think step by step',
    'A person looks into the images and find the <obj> to be stunning. Describe the context and surroundings of <obj> in the picture. Let\'s think step by step'
]

def get_response(image_path, text):
    image = Image.open(image_path)
    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": text}
    ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=150)
    return processor.decode(output[0])

if __name__ == '__main__':
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
    split_path = args.file_path
    image_dir = '/fs/nexus-scratch/aseth125/val2014'
    content_output = []
    content = []
    with open(split_path) as file:
        for lin in file:
            content.append(json.loads(lin))
        
        for data in tqdm.tqdm(content):
            image_path = os.path.join(data["image"])
            obj = data["object"]
            
            for i, prompt in enumerate(LOCALIZATION_PROMPTS):
                data[f'query_{i+1}'] = prompt.replace('<obj>', obj)
                data[f'output_{i+1}'] = get_response(image_path, data[f'query_{i+1}'])
            
            content_output.append(data)

    with open(f'loc_response_{args.split_num}_'+split_path.split('/')[-1],'w') as file:
        for data in content_output:
            json.dump(data, file)
            file.write('\n')

