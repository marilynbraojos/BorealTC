import os
import torch
import matplotlib.pyplot as plt
from borealtc import BorealTC, SlidingWindowDataset
from utils.models import MambaClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

OUT_DIR = "braojos_inference"
os.makedirs(OUT_DIR, exist_ok=True)

dataset = BorealTC("data/borealtc")
window_ds = SlidingWindowDataset(dataset, window_size=170, step_size=50)

sample = window_ds[0]    # get the first window in the dataset
window = sample["window"]
run_id = sample["run_id"]
class_name = sample["class_name"]

print(f"Loaded window | Run: {run_id} | True class: {class_name}")

ckpt_path = "checkpoints/mamba_borealtc.ckpt"
ckpt = torch.load(ckpt_path, map_location=device)
hparams = ckpt["hyper_parameters"]

model = MambaClassifier(
    d_state=hparams["d_state"],
    d_conv=hparams["d_conv"],
    expand=hparams["expand"],
    d_model=hparams["d_model"],
    num_classes=hparams["num_classes"]
).to(device)

model.load_state_dict(ckpt["state_dict"])
model.eval()

x = torch.tensor(window).unsqueeze(0).float().to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    predicted_idx = torch.argmax(probs, dim=1).item()

predicted_name = dataset.classes[predicted_idx]

print(f"PREDICTED CLASS: {predicted_name}")

with open(os.path.join(OUT_DIR, "prediction.txt"), "w") as f:
    f.write(f"Predicted class: {predicted_name}\n")
    f.write(f"True class: {class_name}\n")
    f.write(f"Run ID: {run_id}\n")

print("Saved prediction to braojos_inference/prediction.txt")

plt.figure(figsize=(8,5))
plt.bar(dataset.classes, probs.cpu().numpy()[0])
plt.title(f"Mamba Prediction: {predicted_name}")
plt.ylabel("Probability")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "probabilities.png"))
plt.close()

print("Saved probability plot to braojos_inference/probabilities.png")

plt.figure(figsize=(10,5))
plt.plot(window[:, 0], label="IMU X")
plt.plot(window[:, 1], label="IMU Y")
plt.plot(window[:, 2], label="IMU Z")
plt.title("Raw IMU window used for inference")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "imu_window.png"))
plt.close()

print("Saved IMU window plot to braojos_inference/imu_window.png")