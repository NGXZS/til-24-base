import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
from urllib.request import urlretrieve
from os import remove
import json
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from torch import nn, optim
from fastapi import FastAPI, UploadFile, File
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import json
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
    def __len__(self):
        return len(self.inputs['input_ids'])
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        return item

def load_image(path_or_url):
    """Loads an image from a given URL or path. If the input is a URL,
    it downloads the image and saves it as a temporary file. If the input is a path,
    it loads the image from the path. The image is then converted to RGB format and returned.
    """
    if path_or_url.startswith("http"):  # assume URL if starts with http
        urlretrieve(path_or_url, "imgs/tmp.jpg")
        img = cv2.imread("imgs/tmp.jpg")
        remove("imgs/tmp.jpg")  # cleanup temporary file
    else:
        img = cv2.imread(path_or_url)
    return img

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    return data

def create_annotations_list(filename):
    data = read_jsonl(filename)
    annotations_list = []
    for entry in data:
        image = entry['image']
        annotations = entry['annotations']
        annotations_list.append({'image': image, 'annotations': annotations})
    return annotations_list

annotations_list = create_annotations_list("vlm_sample.jsonl")
entry = annotations_list[1]
dataset = []
for entry in annotations_list:
    curr_image = load_image(f"sample_images/{entry['image']}")
    curr_annotations = entry['annotations']
    for curr_annotation in curr_annotations:
        curr_caption = curr_annotation['caption']
        curr_bbox = curr_annotation['bbox']
        x1, y1, width, height = curr_bbox
        x2, y2 = x1 + width, y1 + height
        cropped_image = curr_image[y1:y2, x1:x2]
        dataset.append((cropped_image, curr_caption))

## CLIP HuggingFace model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

## Preproces data
def preprocess_data(dataset, processor):
    images = [entry[0] for entry in dataset]
    captions = [entry[1] for entry in dataset]
    
    inputs = processor(text=captions, images=images, return_tensors="pt", padding=True)
    return inputs

inputs = preprocess_data(dataset, processor)

train_dataset = CustomDataset(inputs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        
        batch = {k: v.squeeze().to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        
        labels = torch.arange(logits_per_image.size(0), device=device)
        loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.squeeze().to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            labels = torch.arange(logits_per_image.size(0), device=device)
            loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

'''
plt.imshow(img[..., ::-1])  # Image with RGB
plt.axis('off')
plt.show()
'''