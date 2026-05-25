import torch
from transformers import AutoTokenizer
from PIL import Image 
from huggingface_hub import notebook_login
notebook_login()


#general rule for working with dinov3 : 
#1. embed
#class token(globally) or patch token (granular)
#2. visuzalise 
#3. copute isilarity search 
#4. sort by similarity

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
model.eval()

image = Image.open("")

# Process and run
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get embeddings
cls_embedding = outputs.last_hidden_state[:, 0, :]  # whole image embedding
patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # per-patch embeddings

print(cls_embedding.shape)  # e.g. torch.Size([1, 1024])