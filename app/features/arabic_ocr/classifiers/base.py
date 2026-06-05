from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
from arabic_ocr.config import TOP_K
from arabic_ocr.segment.dots import Dot

class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, char_img, dot_list = None):
        """Classify a single character image. Returns a top-K list of (char, confidence) pairs."""

    @abstractmethod
    def predict_batch(self, char_imgs, dot_lists = None):
        """Classify a batch of character images.Returns one top-K list per image."""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier. X: feature vectors, y: correct arabic characters."""

    @abstractmethod
    def save(self, path: Path):
        """save model to disk."""

    @abstractmethod
    def load(self, path: Path):
        """load model from disk."""
