from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
import tqdm
import argparse
# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
COUNTERFACTUAL_PROMPTS = [
    "How would the scene change if the <obj> were invisible?. Let's think step by step",
    "What might replace the <obj> if it vanished from the image?. Let's think step by step",
]

LOCALIZATION_PROMPTS = [
    'Describe where is <obj> in the image. Let\'s think step by step',
    'Examine the presence and location of <obj> in the image. Let\'s think step by step'
]

VISUAL_CONTEXT_PROMPTS = [
    'Interpret the object <obj> with respect to its context within the image. Let\'s think step by step',
    'Analyze the neighboring elements of <obj> in the image. Let\'s think step by step',
]

IDENTIFICATION_PROMPTS = [
    'Is the <obj> present in the image? Let\'s think step by step',
    'Determine whether a <obj> is visible in the image. Let\'s think step by step',
]

LOCALIZATION_PROMPTS_Xray = [
    'Examine the X-ray and identify the region associated with detecting <obj>. Provide a deteministic and concise answer.',
    'Inspect the X-ray and specify the area corresponding to <obj>. Provide a deteministic and concise answer.',
]

IDENTIFICATION_PROMPTS_Xray = [
    'Given this X-ray, identify if the person has <obj>. Provide crisp answer',
    'Based on this X-ray, determine whether the person has <obj>. Provide crisp answer',
]
# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")


# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

def get_response(img_path, text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    #print(output_text)
    return output_text[0]


if __name__ == '__main__':
    flag = 'normal'
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
            if flag == 'med':
                image_path = data["image"]
                obj = data["object"]
            else:
                image_path = os.path.join(image_dir, data["image"])
                obj = ' '.join(data["text"].split(' ')[3:-3])
            
            
            for i, prompt in enumerate(COUNTERFACTUAL_PROMPTS):
                data[f'query_{i+1}'] = prompt.replace('<obj>', obj)
                data[f'output_{i+1}'] = get_response(image_path, data[f'query_{i+1}'])
            
            content_output.append(data)

    with open(f'cot_count_response_{args.split_num}_'+split_path.split('/')[-1],'w') as file:
        for data in content_output:
            json.dump(data, file)
            file.write('\n')