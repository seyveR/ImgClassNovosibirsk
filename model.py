import torch
from torchvision.models import resnext101_32x8d
from torch import nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torchvision.transforms import PILToTensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n', task='detect')

test_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():
    model = resnext101_32x8d().to(device=device)
    model.fc = nn.Linear(model.fc.in_features, 3)
    # optimizer = TheOptimizerClass(*args, **kwargs)

    checkpoint = torch.load('for_three_categories_resnext101_32x8d.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model


def tensor_from_images(image):
    # print(f'url = {image}')
    # image = Image.open(image)
    # print(f'image = {image}')
    image = test_transform(image=np.array(image))['image'].to(device).unsqueeze(0)
    # print(image)
    # image = np.array(image)
    # image = torch.tensor(image)
    return image

def paint_boxes(image_file):
    image = Image.open(image_file)
    # image = np.copy(result.orig_img)
    result = model.predict(image, show_labels=False)[0]
    image_box = np.copy(image)
    for box in result.boxes.xyxy:
        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])
        cv2.rectangle(image_box, (xA, yA), (xB, yB), (0, 0, 255), 5)
    return image_box
