import torch
import gc
import numpy as np 
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    # comment to run on CPU.
    # device_map={"": "cuda"}
    device_map='cuda'
)


def generate_image_caption(image_file_name: str) -> str:
    image = Image.open(image_file_name)
    encoded_image = model.encode_image(image)
    result = model.caption(encoded_image)['caption']
    del encoded_image
    return result

def clean_up():
    """
    Free up GPU memory and run garbage collection.
    """
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    image_file_name = "workflow_video_processing.png"
    caption = generate_image_caption(image_file_name)
    print(caption)