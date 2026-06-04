import torch
import pickle

import cv2
import numpy as np

from app.features.text_extraction.config import parameters
from app.features.text_extraction.train.train_cnn import CNN


class CNNClassifier:
    def __init__(self):
        with open(parameters["CNN"]["encoder_path"], "rb") as f:
            self._encoder = pickle.load(f)
        
        num_classes = len(self._encoder.classes_)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = CNN(num_classes).to(self._device)
        self._model.load_state_dict(torch.load(parameters["CNN"]["model_path"], map_location=self._device))
        self._model.eval()
    
    def _normalize(self, char_img):
        """Resize and center the character image to fixed size"""
        h, w = char_img.shape
        if h == 0 or w == 0:
            return np.ones((parameters["CNN"]["norm_size"], parameters["CNN"]["norm_size"]), dtype=np.float32) * 255
        
        scale = (parameters["CNN"]["norm_size"] - 4) / max(h, w)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        scaled = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.ones((parameters["CNN"]["norm_size"], parameters["CNN"]["norm_size"]), dtype=np.float32) * 255
        y_off = (parameters["CNN"]["norm_size"] - new_h) // 2
        x_off = (parameters["CNN"]["norm_size"] - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
        
        return canvas
    
    def _preprocess(self, char_img):
        """Preprocess single image for CNN input"""
        norm = self._normalize(char_img)
        tensor = torch.FloatTensor(norm).unsqueeze(0).unsqueeze(0) / 255.0
        return tensor.to(self._device)
    
    def predict(self, char_img):
        tensor = self._preprocess(char_img)
        with torch.no_grad():
            outputs = self._model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        top_idx = np.argsort(probs)[::-1][:parameters["CNN"]["TOP_K"]]
        return [(self._encoder.inverse_transform([i])[0], float(probs[i])) for i in top_idx]
    
    def predict_best(self, char_img):
        candidates = self.predict(char_img)
        return candidates[0]
    
    def predict_batch(self, char_imgs):
        """Process batch of images with different sizes"""
        normalized = []
        for img in char_imgs:
            norm = self._normalize(img)
            normalized.append(norm)
        
        batch = np.array(normalized)
        batch = torch.FloatTensor(batch).unsqueeze(1) / 255.0
        batch = batch.to(self._device)
        
        with torch.no_grad():
            outputs = self._model(batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        results = []
        for row in probs:
            top_idx = np.argsort(row)[::-1][:parameters["CNN"]["TOP_K"]]
            results.append([(self._encoder.inverse_transform([i])[0], float(row[i])) for i in top_idx])
        return results