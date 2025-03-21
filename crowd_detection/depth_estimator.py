import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class DepthEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_depth_map(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth_map = self.model(input_tensor)

        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

        # Normalize depth (0 = far, 1 = close)
        return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
