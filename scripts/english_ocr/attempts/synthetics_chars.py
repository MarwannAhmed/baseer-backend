import os
import cv2
import numpy as np
import string
from tqdm import tqdm

OUTPUT_PATH = "data/synthetic_chars/"
SAMPLES_PER_CHAR = 500
IMAGE_SIZE = 32


def add_blur(img, prob=0.3):
    if np.random.random() < prob:
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0.5)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img


def add_thickness_variation(img):
    kernel = np.ones((2, 2), np.uint8)
    if np.random.random() > 0.5:
        img = cv2.erode(img, kernel, iterations=1)
    if np.random.random() > 0.7:
        img = cv2.dilate(img, kernel, iterations=1)
    return img


def add_rotation(img, max_angle=5):
    if np.random.random() > 0.4:
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    return img


def add_shift(img, max_shift=2):
    if np.random.random() > 0.5:
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        shift_y = np.random.randint(-max_shift, max_shift + 1)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
    return img


def add_thinning(img):
    if np.random.random() > 0.6:
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
    return img


def add_break_artifacts(img):
    if np.random.random() > 0.85:
        h, w = img.shape
        y = np.random.randint(1, h-1)
        x = np.random.randint(1, w-1)
        img[y-1:y+1, x-1:x+1] = 255
    return img


def add_uneven_stroke(img):
    if np.random.random() > 0.7:
        kernel_size = np.random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if np.random.random() > 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)
    return img


def add_slight_warp(img):
    if np.random.random() > 0.8:
        h, w = img.shape
        pts1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        shift = np.random.randint(-2, 3)
        pts2 = np.float32([
            [shift, shift],
            [w-1-shift, shift],
            [shift, h-1-shift],
            [w-1+shift, h-1+shift]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h), borderValue=255)
    return img


def create_font_image(char, size=32):
    img = np.ones((size, size), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = np.random.uniform(0.65, 0.95)
    thickness = np.random.randint(1, 3)
    
    text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
    text_x = (size - text_size[0]) // 2 + np.random.randint(-2, 3)
    text_y = (size + text_size[1]) // 2 + np.random.randint(-2, 3)
    
    cv2.putText(img, char, (text_x, text_y), font, font_scale, 0, thickness)
    
    return img


def create_imperfect_character(char, target_size=32):
    img = create_font_image(char, target_size)
    
    img = add_blur(img, prob=0.35)
    img = add_rotation(img, max_angle=4)
    img = add_shift(img, max_shift=1)
    img = add_thickness_variation(img)
    
    if np.random.random() > 0.75:
        img = add_thinning(img)
    
    if np.random.random() > 0.85:
        img = add_break_artifacts(img)
    
    if np.random.random() > 0.8:
        img = add_uneven_stroke(img)
    
    if np.random.random() > 0.9:
        img = add_slight_warp(img)
    
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    return img


def create_dataset(output_path, samples_per_char=500):
    os.makedirs(output_path, exist_ok=True)
    
    digits = list("0123456789")
    uppercase = list(string.ascii_uppercase)
    lowercase = list(string.ascii_lowercase)
    all_chars = digits + uppercase + lowercase
    
    for idx, char in enumerate(tqdm(all_chars, desc="Generating characters")):
        sample_num = idx + 1
        folder_name = f"Sample{sample_num:03d}"
        folder_path = os.path.join(output_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        for i in range(samples_per_char):
            img = create_imperfect_character(char)
            img_resized = cv2.resize(img, (32, 32))
            
            filename = f"{folder_name}_{i+1:03d}.png"
            filepath = os.path.join(folder_path, filename)
            cv2.imwrite(filepath, img_resized)
        
        if (idx + 1) % 10 == 0:
            print(f"  Generated {sample_num} characters: {char}")
    
    print(f"\nDataset created at: {output_path}")
    print(f"Total folders: {len(all_chars)}")
    print(f"Total images: {len(all_chars) * samples_per_char}")
    
    with open(os.path.join(output_path, "README.txt"), "w") as f:
        f.write("Synthetic Character Dataset (No Noise Dots)\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples per character: {samples_per_char}\n")
        f.write(f"Total characters: {len(all_chars)}\n")
        f.write(f"Total images: {len(all_chars) * samples_per_char}\n\n")
        
        f.write("Mapping:\n")
        f.write(f"Sample001-Sample010: Digits 0-9\n")
        f.write(f"Sample011-Sample036: Uppercase A-Z\n")
        f.write(f"Sample037-Sample062: Lowercase a-z\n")
        
        f.write("\nImperfections applied (no salt/pepper noise):\n")
        f.write("- Gaussian blur (simulates slight defocus)\n")
        f.write("- Rotation (±4°)\n")
        f.write("- Pixel shifts (±1px)\n")
        f.write("- Thickness variation (erosion/dilation)\n")
        f.write("- Stroke thinning\n")
        f.write("- Break artifacts (small missing sections)\n")
        f.write("- Uneven stroke weight\n")
        f.write("- Slight perspective warp\n")


def visualize_samples(output_path, num_samples=5):
    import matplotlib.pyplot as plt
    
    folders = [d for d in os.listdir(output_path) if d.startswith('Sample')][:10]
    
    fig, axes = plt.subplots(len(folders), num_samples, figsize=(num_samples * 2, len(folders) * 2))
    
    for i, folder in enumerate(folders):
        folder_path = os.path.join(output_path, folder)
        images = [f for f in os.listdir(folder_path) if f.endswith('.png')][:num_samples]
        
        for j, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
        
        axes[i, 0].set_ylabel(folder, rotation=90, size=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "sample_visualization.png"))
    plt.show()


if __name__ == "__main__":
    create_dataset(OUTPUT_PATH, samples_per_char=500)
    print("\nTo visualize samples, run: visualize_samples(OUTPUT_PATH)")