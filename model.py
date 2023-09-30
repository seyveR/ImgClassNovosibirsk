import torch
from torchvision.models import resnext101_32x8d
from torch import nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torchvision.transforms import PILToTensor
import albumentations as A
from albumentations.pytorch import ToTensorV2


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

    checkpoint = torch.load('three_cats_norm.pth', map_location=device)
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