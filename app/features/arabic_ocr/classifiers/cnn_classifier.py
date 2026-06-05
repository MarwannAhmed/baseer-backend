from pathlib import Path
from typing import Optional
import numpy as np
from arabic_ocr.config import TOP_K, NORM_SIZE
from arabic_ocr.segment.dots import Dot
from arabic_ocr.features.normalize import normalize
from arabic_ocr.features.dot_features import dot_features
from .base import BaseClassifier
import torch
import torch.nn.functional as F

class CNNClassifier(BaseClassifier):

    def __init__(self, n_classes = 28, lr = 1e-3, epochs = 40):
        self.n_classes = n_classes
        self.lr        = lr
        self.epochs    = epochs
        self.model     = None
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class ResBlock(nn.Module):
            def __init__(self, in_planes, out_planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_planes)
                self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_planes)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != out_planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_planes)
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out

        class _Net(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.in_planes = 64

                self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                
                # ResNet layers
                self.layer1 = self._make_layer(ResBlock, 64,  2, stride=1)
                self.layer2 = self._make_layer(ResBlock, 128, 2, stride=2)
                self.layer3 = self._make_layer(ResBlock, 256, 2, stride=2)
                
                self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
                self.fc1 = nn.Linear(256 * 2 * 2, 512)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(512 + 4, n_classes)  # +4 for dot_features

            def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for s in strides:
                    layers.append(block(self.in_planes, planes, s))
                    self.in_planes = planes
                return nn.Sequential(*layers)

            def forward(self, x, dots):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                
                out = self.dropout(F.relu(self.fc1(out)))
                out = torch.cat([out, dots], dim=1)
                return self.fc2(out)

        return _Net(self.n_classes)

    def train(self, X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import LabelEncoder

        self._encoder = LabelEncoder()
        self._encoder.fit(y)
        self.classes_ = self._encoder.classes_
        self.n_classes = len(self.classes_)
        y_idx = self._encoder.transform(y)

        imgs  = torch.tensor(X / 255.0, dtype=torch.float32).unsqueeze(1) 
        d_feat = torch.zeros(len(X), 4, dtype=torch.float32)
        labels = torch.tensor(y_idx, dtype=torch.long)

        self.model = self._build_model()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(imgs, d_feat, labels), batch_size=64, shuffle=True
        )

        self.model.train()
        for _ in range(self.epochs):
            for xb, db, yb in loader:
                opt.zero_grad()
                criterion(self.model(xb, db), yb).backward()
                opt.step()
            scheduler.step()

    def predict(self, character_image, dot_list = None):
        return self.predict_batch([character_image], [dot_list])[0]

    def predict_batch(self, character_images, dot_lists = None):
        if dot_lists is None:
            dot_lists = [None] * len(character_images)

        norm_imgs = np.stack([normalize(img) for img in character_images])
        imgs_t = torch.tensor(norm_imgs / 255.0, dtype=torch.float32).unsqueeze(1)
        dots_t = torch.tensor(
            np.stack([dot_features(d) for d in dot_lists]),
            dtype=torch.float32,
        )
        self.model.eval()
        with torch.no_grad():
            logits = self.model(imgs_t, dots_t)
            proba  = F.softmax(logits, dim=1).numpy()

        return [self._top_k(row) for row in proba]

    def _top_k(self, proba):
        k = min(TOP_K, len(proba))
        indices = np.argsort(proba)[::-1][:k]
        return [(str(self.classes_[i]), float(proba[i])) for i in indices]

    def save(self, path):
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "classes":     self.classes_,
            "n_classes":   self.n_classes,
        }, path)

    def load(self, path):
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.n_classes = data["n_classes"]
        self.classes_  = data["classes"]
        self.model     = self._build_model()
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
