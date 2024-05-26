import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.load_state_dict(torch.load('clip_model_weights.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def predict_bbox(image_path, caption):
    image = load_image(image_path)
    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
    predicted_bbox = [50, 50, 100, 100]
    return predicted_bbox

image_path = input("Insert image: ")
caption = input("Insert caption: ")
bbox = predict_bbox(image_path, caption)

# Display the image with the predicted bounding box
image = load_image(image_path)
plt.imshow(image)
rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                     edgecolor='red', facecolor='none', linewidth=2)
plt.gca().add_patch(rect)
plt.axis("off")
plt.show()