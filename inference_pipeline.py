from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import densenet121


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def discover_class_names(data_train_dir: str = "data/train") -> list[str] | None:
    """Return sorted class folder names from data/train if available."""
    class_root = Path(data_train_dir)
    if not class_root.is_absolute():
        class_root = Path(__file__).resolve().parent / class_root
    if not class_root.exists():
        return None

    class_names = sorted(item.name for item in class_root.iterdir() if item.is_dir())
    return class_names or None


class DenseNet121InferencePipeline:
    def __init__(
        self,
        model_path: str = "best_densenet122.pth",
        class_names: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).resolve().parent / self.model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_dict = torch.load(self.model_path, map_location="cpu")
        if not isinstance(self.state_dict, dict):
            raise TypeError("Checkpoint must be a state_dict dictionary.")

        self.num_classes = self._infer_num_classes_from_state_dict(self.state_dict)
        self.class_names = class_names or discover_class_names() or [
            f"class_{idx}" for idx in range(self.num_classes)
        ]
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "Class names count does not match classifier output size. "
                f"class_names={len(self.class_names)}, num_classes={self.num_classes}"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.model = self._build_model()

    def _infer_num_classes_from_state_dict(self, state_dict: dict[str, Any]) -> int:
        classifier_weight_key = "classifier.weight"
        if classifier_weight_key not in state_dict:
            raise KeyError(
                f"Expected '{classifier_weight_key}' in state_dict, found keys like: "
                f"{list(state_dict.keys())[:5]}"
            )
        return int(state_dict[classifier_weight_key].shape[0])

    def _build_model(self) -> nn.Module:
        model = densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, self.num_classes)
        model.load_state_dict(self.state_dict)
        model.to(self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(image_tensor)
        probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()

        predicted_index = int(torch.argmax(probabilities_tensor).item())
        confidence = float(probabilities_tensor[predicted_index].item())
        probabilities = {
            class_name: float(probabilities_tensor[idx].item())
            for idx, class_name in enumerate(self.class_names)
        }

        return {
            "predicted_index": predicted_index,
            "predicted_class": self.class_names[predicted_index],
            "confidence": confidence,
            "probabilities": probabilities,
        }
