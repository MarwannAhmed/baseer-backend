import json
import time
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timm
import yaml

from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss

COCO_ROOT = Path("/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017")
TRAIN_FOLDER = COCO_ROOT / "train2017"
VAL_FOLDER = COCO_ROOT / "val2017"
TRAIN_ANNOTATIONS = COCO_ROOT / "annotations" / "instances_train2017.json"
VAL_ANNOTATIONS = COCO_ROOT / "annotations" / "instances_val2017.json"

WORK = Path("/kaggle/working")
TRAIN_LABEL_FOLDER = WORK / "coco_labels" / "train2017"
VAL_LABEL_FOLDER = WORK / "coco_labels" / "val2017"
COCO_YAML = WORK / "coco_local.yaml"

(WORK / "models").mkdir(parents=True, exist_ok=True)
(WORK / "results").mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 640
EPOCHS = 15
BATCH = 8
LEARNING_RATE = 1e-3
NUM_CLASSES = 80
SPEED_RUNS = 20
CHECKPOINT_SAVE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

P3_CHANNELS = 64
P4_CHANNELS = 128
P5_CHANNELS = 256

COCO_CAT_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}

BACKBONE_NAME = "mobilenetv2_100"
OUT_INDICES = (2, 3, 4)

PT_MODEL_PATH = str(WORK / "models/backbone.pt")
ONNX_MODEL_PATH = str(WORK / "models/backbone.onnx")


# converting COCO JSON to text format for YOLO compatibility
def json_to_yolo(annotation_json, label_folder):
    label_folder.mkdir(parents=True, exist_ok=True)
    existing = list(label_folder.glob("*.txt"))
    if len(existing) > 100:
        return
    with open(annotation_json) as f:
        data = json.load(f)
    image_info = {
        image["id"]: (image["file_name"], image["width"], image["height"])
        for image in data["images"]
    }
    annotations_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        if (
            annotation.get("iscrowd", 0) == 0
            and annotation["category_id"] in COCO_CAT_MAP
        ):
            annotations_by_image[annotation["image_id"]].append(annotation)
    for image_id, (file_name, image_width, image_height) in image_info.items():
        stem = Path(file_name).stem
        label_path = label_folder / f"{stem}.txt"
        anns = annotations_by_image.get(image_id, [])
        with open(label_path, "w") as f:
            for annotation in anns:
                x1, y1, bw, bh = annotation["bbox"]
                cx = max(0.0, min(1.0, (x1 + bw / 2) / image_width))
                cy = max(0.0, min(1.0, (y1 + bh / 2) / image_height))
                nw = max(0.0, min(1.0, bw / image_width))
                nh = max(0.0, min(1.0, bh / image_height))
                class_index = COCO_CAT_MAP[annotation["category_id"]]
                f.write(f"{class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


json_to_yolo(TRAIN_ANNOTATIONS, TRAIN_LABEL_FOLDER)
json_to_yolo(VAL_ANNOTATIONS, VAL_LABEL_FOLDER)

coco_yaml_content = {
    "path": str(COCO_ROOT),
    "train": str(TRAIN_FOLDER),
    "val": str(VAL_FOLDER),
    "nc": 80,
    "names": {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    },
}

with open(COCO_YAML, "w") as file:
    yaml.dump(coco_yaml_content, file, default_flow_style=False, sort_keys=False)


class YOLOFormatDataset(Dataset):
    def __init__(self, image_folder, label_folder, image_size=IMAGE_SIZE):
        self.image_paths = sorted(Path(image_folder).glob("*.jpg"))
        self.label_folder = Path(label_folder)
        self.image_size = image_size
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_folder}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        label_path = self.label_folder / (path.stem + ".txt")
        image = cv2.imread(str(path))
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes = []
        if label_path.exists():
            with open(label_path) as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append(list(map(float, parts)))
        labels = (
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        )
        return image, labels


def build_detection_batch(batch):
    images, labels_per_image = zip(*batch)
    stacked_images = torch.stack(images, dim=0)

    all_classes = []
    all_bounding_boxes = []
    all_batch_indices = []

    for image_index, labels in enumerate(labels_per_image):
        if len(labels) == 0:
            continue

        all_classes.append(labels[:, 0:1])
        all_bounding_boxes.append(labels[:, 1:5])

        batch_indices = torch.full(
            (len(labels),),
            image_index,
            dtype=torch.float32,
        )
        all_batch_indices.append(batch_indices)

    return {
        "img": stacked_images,
        "cls": (torch.cat(all_classes, dim=0) if all_classes else torch.zeros((0, 1))),
        "bboxes": (
            torch.cat(all_bounding_boxes, dim=0)
            if all_bounding_boxes
            else torch.zeros((0, 4))
        ),
        "batch_idx": (
            torch.cat(all_batch_indices, dim=0) if all_batch_indices else torch.zeros(0)
        ),
    }


def make_adapter(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class HybridModel(nn.Module):
    def __init__(self, backbone_name, out_indices, nc=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )
        with torch.no_grad():
            features = self.backbone(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        backbone_channels = [feature.shape[1] for feature in features]

        self.adapter_p3 = make_adapter(backbone_channels[0], P3_CHANNELS)
        self.adapter_p4 = make_adapter(backbone_channels[1], P4_CHANNELS)
        self.adapter_p5 = make_adapter(backbone_channels[2], P5_CHANNELS)
        self.sppf = SPPF(P5_CHANNELS, P5_CHANNELS, k=5)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_p4_fusion = C2f(
            P5_CHANNELS + P4_CHANNELS, P4_CHANNELS, n=1, shortcut=False
        )
        self.top_down_p3_fusion = C2f(
            P4_CHANNELS + P3_CHANNELS, P3_CHANNELS, n=1, shortcut=False
        )
        self.down_p3 = Conv(P3_CHANNELS, P3_CHANNELS, k=3, s=2)
        self.bottom_up_p4_fusion = C2f(
            P3_CHANNELS + P4_CHANNELS, P4_CHANNELS, n=1, shortcut=False
        )
        self.down_p4 = Conv(P4_CHANNELS, P4_CHANNELS, k=3, s=2)
        self.bottom_up_p5_fusion = C2f(
            P4_CHANNELS + P5_CHANNELS, P5_CHANNELS, n=1, shortcut=False
        )
        self.detect = Detect(nc=nc, ch=(P3_CHANNELS, P4_CHANNELS, P5_CHANNELS))
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.bias_init()

    def forward(self, x):
        backbone_features = self.backbone(x)

        p3_features = self.adapter_p3(backbone_features[0])
        p4_features = self.adapter_p4(backbone_features[1])
        p5_features = self.adapter_p5(backbone_features[2])

        p5_enhanced = self.sppf(p5_features)
        p4_fused = self.top_down_p4_fusion(
            torch.cat([self.upsample(p5_enhanced), p4_features], dim=1)
        )
        p3_output = self.top_down_p3_fusion(
            torch.cat([self.upsample(p4_fused), p3_features], dim=1)
        )
        p4_output = self.bottom_up_p4_fusion(
            torch.cat([self.down_p3(p3_output), p4_fused], dim=1)
        )
        p5_output = self.bottom_up_p5_fusion(
            torch.cat([self.down_p4(p4_output), p5_enhanced], dim=1)
        )

        return self.detect([p3_output, p4_output, p5_output])


class DetectionModelWrapper(nn.Module):
    def __init__(self, hybrid):
        super().__init__()
        self._hybrid = hybrid
        self.model = nn.ModuleList([hybrid, hybrid.detect])
        self.stride = hybrid.detect.stride
        self.nc = NUM_CLASSES
        self.names = {i: str(i) for i in range(NUM_CLASSES)}
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    def forward(self, x):
        return self._hybrid(x)

    def parameters(self, recurse=True):
        return self._hybrid.parameters(recurse=recurse)


def init_pretrained_head(model, yolov8n_path=str(WORK / "models/yolov8n.pt")):
    if not Path(yolov8n_path).exists():
        return
    pretrained_weights = YOLO(yolov8n_path).model.state_dict()
    detection_head_weights = {
        parameter_name.replace("model.22.", ""): parameter_value
        for parameter_name, parameter_value in pretrained_weights.items()
        if parameter_name.startswith("model.22.")
    }
    try:
        model.detect.load_state_dict(
            detection_head_weights,
            strict=True,
        )
    except RuntimeError:
        print("Cannot detect YOLOv8n head, training from random initialising point.")


def train(wrapper, dataloader, n_epochs, save_pt, save_every=CHECKPOINT_SAVE):
    loss_fn = v8DetectionLoss(wrapper)
    optimizer = optim.SGD(
        wrapper.parameters(),
        lr=LEARNING_RATE,
        momentum=0.937,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    checkpoint_folder = WORK / f"models/checkpoint"
    checkpoint_folder.mkdir(exist_ok=True)

    existing_checkpoints = sorted(checkpoint_folder.glob("epoch_*.pt"))
    if existing_checkpoints:
        latest = existing_checkpoints[-1]
        checkpoint = torch.load(latest, map_location=DEVICE)
        wrapper._hybrid.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        for _ in range(start_epoch):
            scheduler.step()

    loss_history = []
    wrapper.train()

    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()

        for batch in dataloader:
            batch["img"] = batch["img"].to(DEVICE)
            batch["cls"] = batch["cls"].to(DEVICE)
            batch["bboxes"] = batch["bboxes"].to(DEVICE)
            batch["batch_idx"] = batch["batch_idx"].to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                predictions = wrapper(batch["image"])
                result = loss_fn(predictions, batch)
                loss = result[0] if isinstance(result, tuple) else result
                if loss.ndim > 0:
                    loss = loss.sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(wrapper.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 100 == 0:
                print(
                    f"Epoch {epoch+1} — Batch {n_batches}/{len(dataloader)}: loss={epoch_loss/n_batches:.4f}",
                    flush=True,
                )

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        secs = time.time() - start_time
        loss_history.append(avg)
        print(f"Epoch {epoch+1:3d}/{n_epochs}: loss={avg:.4f}, time={secs/60:.1f}min")

        if (epoch + 1) % save_every == 0 or epoch == n_epochs - 1:
            checkpoint_path = checkpoint_folder / f"epoch_{epoch+1:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": wrapper._hybrid.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path.name}")

    torch.save(wrapper._hybrid.state_dict(), save_pt)
    print(f"Final model saved: {save_pt}")
    return loss_history


def export_onnx(model, path):
    model.eval().cpu()
    model.detect.export = True
    torch.onnx.export(
        model,
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        path,
        opset_version=12,
        dynamo=False,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
    )
    model.detect.export = False


dataset = YOLOFormatDataset(str(TRAIN_FOLDER), str(TRAIN_LABEL_FOLDER))
dataloader = DataLoader(
    dataset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=4,
    collate_fn=build_detection_batch,
    pin_memory=True,
)

hybrid = HybridModel(
    BACKBONE_NAME,
    OUT_INDICES,
    nc=NUM_CLASSES,
).to(DEVICE)

total_parameters = sum(parameter.numel() for parameter in hybrid.parameters()) / 1e6
print(f"Parameters: {total_parameters:.1f}M")

init_pretrained_head(hybrid)

wrapper = DetectionModelWrapper(hybrid).to(DEVICE)

if Path(PT_MODEL_PATH).exists():
    hybrid.load_state_dict(torch.load(PT_MODEL_PATH, map_location=DEVICE))
else:
    print(f"Starting training: {EPOCHS} epochs.")

    loss_history = train(
        wrapper,
        dataloader,
        EPOCHS,
        PT_MODEL_PATH,
    )

    plt.figure(figsize=(8, 3))
    plt.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        color="#0000FF",
        lw=2,
        marker="o",
        ms=4,
    )
    plt.title("Detection Training Loss", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("v8 Detection Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        str(WORK / "results/loss.png"),
        dpi=130,
    )
    plt.close()

hybrid.to("cpu")
export_onnx(hybrid, ONNX_MODEL_PATH)


del hybrid, wrapper
torch.cuda.empty_cache()
