from PIL import Image as pil_image
from torchvision import transforms
import torch

def predict_image(image, model):
    image = pil_image.open(image).convert("RGB")
    test_transforms = transforms.Compose(
            [
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    image = test_transforms(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
        preds_val = preds.data.squeeze().numpy()
    return preds_val




