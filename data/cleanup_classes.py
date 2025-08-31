import os
import torch
import shutil
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Paths
model_path = "data/data_cleaner/bird_filter_resnet18.pth"
data_dir = Path("data/csv2/frames")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class index map
idx_to_class = {0: 'bird', 1: 'no_bird'}

# Traverse species directories
for species_dir in data_dir.iterdir():
    if not species_dir.is_dir():
        continue

    no_bird_dir = species_dir / "no_bird"
    moved_any = False

    for img_path in list(species_dir.glob("*.jpg")):
        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1).item()

            if idx_to_class[pred_class] == "no_bird":
                if not moved_any:
                    os.makedirs(no_bird_dir, exist_ok=True)
                    moved_any = True
                shutil.move(str(img_path), no_bird_dir / img_path.name)
                print(f"[MOVED] {img_path} → {no_bird_dir / img_path.name}")

        except Exception as e:
            print(f"[ERROR] Failed on {img_path}: {e}")
