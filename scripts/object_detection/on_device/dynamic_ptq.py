import os
import time
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from ultralytics import YOLO

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

IMAGE_SIZE = 640
SPEED_RUNS = 20

non_optimised_pt_path = Path("models_data/backbone_mobilenetv2.pt")
non_optimised_onnx_path = Path("models_data/backbone_mobilenetv2.onnx")
optimised_onnx_path = Path("models_data/backbone_mobilenetv2_int8.onnx")

if not non_optimised_pt_path.exists():
    print("Original non-optimised PT format model not found.")
    exit(1)

if non_optimised_onnx_path.exists():
    print(f"Non-optimised model already exported to ONNX format.")
else:
    model_pt = YOLO(str(non_optimised_pt_path))
    exported = model_pt.export(format="onnx", imgsz=IMAGE_SIZE, simplify=True, opset=12)
    default_export = Path("models/yolov8n.onnx")
    if default_export.exists():
        default_export.rename(non_optimised_onnx_path)
    elif Path("yolov8n.onnx").exists():
        Path("yolov8n.onnx").rename(non_optimised_onnx_path)

non_optimised_size = os.path.getsize(non_optimised_onnx_path) / 1_000_000


def preprocess_image(image_path: str, image_size: int = IMAGE_SIZE) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image


if optimised_onnx_path.exists():
    print(f"Optimised model already constructed and exported to ONNX format.")
else:
    quantize_dynamic(
        model_input=str(non_optimised_onnx_path),
        model_output=str(optimised_onnx_path),
        weight_type=QuantType.QUInt8,
    )

# metadata injection for optimised model
optimised_model = onnx.load(str(optimised_onnx_path))
existing_keys = [prop.key for prop in optimised_model.metadata_props]
names_dictionary = {
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
}
needed = {
    "stride": "32",
    "nc": "80",
    "task": "detect",
    "imgsz": "[640, 640]",
    "batch": "1",
    "names": str(names_dictionary),
}

for key, value in needed.items():
    if key not in existing_keys:
        entry = optimised_model.metadata_props.add()
        entry.key = key
        entry.value = value

onnx.save(optimised_model, str(optimised_onnx_path))

optimised_size = os.path.getsize(optimised_onnx_path) / 1_000_000

# following code is for performance comparisons between non-optimised and optimised model

print(f"Non-optimised ONNX size: {non_optimised_size:.1f} MB")
print(f"Optimised ONNX size: {optimised_size:.1f} MB")
print(f"Size reduction: {non_optimised_size / optimised_size:.1f}x smaller")

test_image = "test_image.jpg"


def onnx_speed_test(onnx_path: str, test_image_path: str, speed_runs: int = SPEED_RUNS):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_path,
        session_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    image = preprocess_image(test_image_path)

    session.run(None, {input_name: image})

    times = []
    for _ in range(speed_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: image})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return sum(times) / len(times)


non_optimised_inference = onnx_speed_test(str(non_optimised_onnx_path), test_image)
print(
    f"Non-optimised model inference: {non_optimised_inference:.1f} ms, {1000/non_optimised_inference:.1f} FPS"
)

optimised_inference = onnx_speed_test(str(optimised_onnx_path), test_image)
print(
    f"Optimised model inference: {optimised_inference:.1f} ms, {1000/optimised_inference:.1f} FPS"
)

print(f"Speedup: {non_optimised_inference / optimised_inference:.2f}x faster")


def map_evaluation(model_path: str):
    model = YOLO(model_path, task="detect")
    metrics = model.val(
        data="coco128.yaml", imgsz=IMAGE_SIZE, verbose=False, plots=False, rect=False
    )
    return float(metrics.box.map50), float(metrics.box.map)


non_optimised_map50, non_optimised_map50to95 = map_evaluation(
    str(non_optimised_onnx_path)
)
print(
    f"Non-optmised model mAP@0.5: {non_optimised_map50:.4f}, percentage: ({non_optimised_map50*100:.1f}%)"
)
print(
    f"Non-optmised model mAP@0.5:0.95: {non_optimised_map50to95:.4f}, percentage: ({non_optimised_map50to95*100:.1f}%)"
)

optimised_map50, optimised_map50to95 = map_evaluation(str(optimised_onnx_path))
print(
    f"Optimised model mAP@0.5: {optimised_map50:.4f}, percentage: ({optimised_map50*100:.1f}%)"
)
print(
    f"Optimised model mAP@0.5:0.95: {optimised_map50to95:.4f}, percentage: ({optimised_map50to95*100:.1f}%)"
)

map50_drop = (non_optimised_map50 - optimised_map50) * 100
print(f"Accuracy drop: {map50_drop:.2f}% mAP@0.5")

results = [
    {
        "Model": "YOLOv8n-FP32-ONNX",
        "Size in MB": round(non_optimised_size, 1),
        "Inference in ms": round(non_optimised_inference, 1),
        "FPS": round(1000 / non_optimised_inference, 1),
        "mAP@0.5": round(non_optimised_map50, 4),
        "mAP@0.5:0.95": round(non_optimised_map50to95, 4),
    },
    {
        "Model": "YOLOv8n-INT8-PTQ",
        "Size in MB": round(optimised_size, 1),
        "Inference in ms": round(optimised_inference, 1),
        "FPS": round(1000 / optimised_inference, 1),
        "mAP@0.5": round(optimised_map50, 4),
        "mAP@0.5:0.95": round(optimised_map50to95, 4),
    },
]

dataframe = pd.DataFrame(results)
ptq_csv = "results/ptq_results.csv"
dataframe.to_csv(ptq_csv, index=False)
print(dataframe.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("PTQ Results Comparison", fontsize=13, fontweight="bold")

colors = ["#0000FF", "#00FF00"]
labels = ["Non-Optimised ONNX", "Optimised ONNX"]
sizes = [non_optimised_size, optimised_size]
speeds = [non_optimised_inference, optimised_inference]
maps50 = [non_optimised_map50 * 100, optimised_map50 * 100]

ax = axes[0]
bars = ax.bar(labels, sizes, color=colors, edgecolor="white", width=0.5)
ax.set_title("Size in MB")
ax.set_ylabel("MB")
ax.set_ylim(0, max(sizes) * 1.35)
for bar, v in zip(bars, sizes):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{v:.1f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

ax = axes[1]
bars = ax.bar(labels, speeds, color=colors, edgecolor="white", width=0.5)
ax.set_title("Inference Time in ms")
ax.set_ylabel("ms / image")
ax.set_ylim(0, max(speeds) * 1.35)
for bar, v in zip(bars, speeds):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{v:.0f}ms",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

ax = axes[2]
bars = ax.bar(labels, maps50, color=colors, edgecolor="white", width=0.5)
ax.set_title("mAP@0.5 in %")
ax.set_ylabel("%")
ax.set_ylim(0, 100)
for bar, v in zip(bars, maps50):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 1,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.tight_layout()
chart_path = "results/ptq_chart.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
