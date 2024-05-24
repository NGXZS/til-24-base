import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchinfo import summary
import urllib
from PIL import Image
from urllib.request import urlretrieve
from os import remove

# 1. Load the pre-trained SSD300 VGG16 model
def load_pretrained_model():
    weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    ssd_model = torchvision.models.detection.ssd300_vgg16(
        weights=weights, box_score_thresh=0.9
    )
    ssd_model.eval()  # Set the model to evaluation mode
    return ssd_model, weights
#summary(ssd_model)

## 2. Load image
def load_image(path_or_url):
    """Loads an image from a given URL or path. If the input is a URL,
    it downloads the image and saves it as a temporary file. If the input is a path,
    it loads the image from the path. The image is then converted to RGB format and returned.
    """
    if path_or_url.startswith("http"):  # assume URL if starts with http
        urlretrieve(path_or_url, "imgs/tmp.jpg")
        img = Image.open("imgs/tmp.jpg").convert("RGB")
        remove("imgs/tmp.jpg")  # cleanup temporary file
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img

#img = load_image("imgs/catdog.webp")

## 3. Preprocess the image
def preprocess_image(img):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    processed_image = transform(img).unsqueeze(0)
    return processed_image

# 4. Label encoding and running interface
def model_interface(ssd_model, processed_image):
    with torch.no_grad():
        detections = ssd_model(processed_image)[0]
    return detections

# 5. Extract and visualise output
def visualise_output(img, detections, weights):
    id_2_label = {idx: x for idx, x in enumerate(weights.meta["categories"])}
    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]
    class_colors = ["red", "blue", "green", "purple", "orange"]

    plt.imshow(img)
    count = 0
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.1:
            count += 1
            class_id = label.item()  # Get the class ID
            class_color = class_colors[
                class_id % len(class_colors)
            ]  # Assign a color based on class ID
            print(id_2_label[label.item()], score.item())
            plt.gca().add_patch(
                plt.Rectangle(
                    # box[0] is x1, box[1] is y1, box[2] is x2, box[3] is y2
                    (box[0], box[1]), 
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    edgecolor=class_color,
                    linewidth=2,
                )
            )
            plt.text(
                box[0],
                box[1],
                f"{id_2_label[label.item()]} ({score:.2f})",
                color=class_color,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )
    plt.axis("off")
    plt.show()

def main(path_or_url):
    ssd_model, weights = load_pretrained_model()
    img = load_image(path_or_url)
    processed_image = preprocess_image(img)
    detections = model_interface(ssd_model, processed_image)
    visualise_output(img, detections, weights)
    
if __name__ == "__main__":
    main('imgs/testing.png')
