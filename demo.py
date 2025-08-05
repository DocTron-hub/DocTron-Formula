import argparse
import json
import os

import torch
from qwen_vl_utils import process_vision_info
from rapidfuzz.distance import Levenshtein
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

user_prompt = "Analyze the image. Extract and output only the LaTeX formulas present in the image, in LaTeX code format. Ignore inline formulas, all other text, and do not include any explanations."


def read_input_file(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    image_path = data[0]['images'][0]
    gt_latex_code = data[0]['messages'][1]['content']
    
    return image_path, gt_latex_code


class ImageProcessor:
    def __init__(self, args):
        self.args = args
        self.model, self.vis_processor = self.load_model_and_processor()
        self.generate_kwargs = dict(
            max_new_tokens=2048,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            repetition_penalty=1.0,
        ) 

    def load_model_and_processor(self):
        # Load model
        checkpoint = self.args.ckpt
        vis_processor = AutoProcessor.from_pretrained(checkpoint)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
        model.eval()

        return model, vis_processor

    def process_single_image(self, image_path):
        question = user_prompt

        try:
            image_local_path = "file://" + image_path

            messages = []
            messages.append(
                {"role": "user", "content": [
                        {"type": "image", "image": image_local_path, "min_pixels": 32 * 32, "max_pixels": 512 * 512},
                        {"type": "text", "text": question},
                    ]
                }
            )

            text = self.vis_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info([messages])

            inputs = self.vis_processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.vis_processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            model_answer = out[0]
        except Exception as e:
            print(e, flush=True)
            model_answer = "None"

        return model_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="DocTron/DocTron-Formula")
    parser.add_argument("--input_file", type=str, default="line-level")
    args = parser.parse_args()

    # Init model
    processor = ImageProcessor(args)

    # Read input file
    input_file = os.path.join('./asset/test_jsons', f"{args.input_file}.json")
    image_path, gt_latex_code = read_input_file(input_file)

    # Inference
    pred_latex_code = processor.process_single_image(image_path)
    print(f'GT:\n{gt_latex_code}')
    print(f'Prediction:\n{pred_latex_code}')

    # Eval
    edit_dist = Levenshtein.normalized_distance(pred_latex_code, gt_latex_code)
    print(f'Edit distance: {edit_dist}')
    