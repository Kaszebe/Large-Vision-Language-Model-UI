import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import argparse
import os

# Use a relative path or environment variable for the model path
model_path = os.environ.get("QWEN_MODEL_PATH", "path/to/model")

def load_model(use_flash_attention=False):
    model_kwargs = {
        "torch_dtype": torch.float16,  # Use float16 for AWQ compatibility
        "device_map": "auto",
    }
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    return model

processor = AutoProcessor.from_pretrained(model_path)

def process_input(image, video, prompt, temperature=0.8, top_k=50, top_p=0.9, max_tokens=100):
    if image is not None:
        media_type = "image"
        media = image
    elif video is not None:
        media_type = "video"
        media = video
    else:
        return "Please upload an image or video."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": media_type, media_type: media},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    response = output_text[0].split("assistant\n")[-1].strip()
    return response

def create_interface():
    interface = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Image(type="filepath", label="Upload Image (optional)"),
            gr.Video(label="Upload Video (optional)"),
            gr.Textbox(label="Text Prompt"),
            gr.Slider(0.1, 1.0, value=0.8, label="Temperature"),
            gr.Slider(1, 100, value=50, step=1, label="Top-k"),
            gr.Slider(0.1, 1.0, value=0.9, label="Top-p"),
            gr.Slider(1, 500, value=100, step=10, label="Max Tokens")
        ],
        outputs=gr.Textbox(label="Generated Description"),
        title="Qwen2-VL-72B Vision-Language Model",
        description="Upload an image or video and enter a prompt to generate a description.",
    )
    return interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen2-VL model with optional Flash Attention 2")
    parser.add_argument("--flash-attn2", action="store_true", help="Use Flash Attention 2")
    args = parser.parse_args()
    
    model = load_model(use_flash_attention=args.flash_attn2)
    interface = create_interface()
    interface.launch(share=True)
