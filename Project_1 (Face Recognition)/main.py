
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def list_videos_recursive(root):
    videos = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                full = os.path.join(dirpath, f)
                person = os.path.basename(os.path.dirname(full))
                videos.append((full, person))
    return videos

# ------------------------------------------------------------
# Step 1: Extract Frames
# ------------------------------------------------------------
def extract_n_frames(video_path, out_folder, n_frames=120):
    ensure_dir(out_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Cannot open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // n_frames)
    saved = 0

    for i in range(0, total_frames, step):
        if saved >= n_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(out_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_img_{saved}.jpg")
        cv2.imwrite(fname, frame)
        saved += 1

    cap.release()
    return saved

# ------------------------------------------------------------
# Step 2: Convert to Greyscale + Binary
# ------------------------------------------------------------
def convert_to_greyscale_and_binary(frames_folder, grey_folder, binary_folder):
    ensure_dir(grey_folder)
    ensure_dir(binary_folder)

    for f in os.listdir(frames_folder):
        if not f.lower().endswith(".jpg"):
            continue
        path = os.path.join(frames_folder, f)
        img = cv2.imread(path)
        if img is None:
            continue
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(grey_folder, f), grey)
        cv2.imwrite(os.path.join(binary_folder, f), binary)

# ------------------------------------------------------------
# Step 3: Crop Faces
# ------------------------------------------------------------
def crop_faces(frames_folder, cropped_folder):
    ensure_dir(cropped_folder)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for f in os.listdir(frames_folder):
        if not f.lower().endswith(".jpg"):
            continue
        path = os.path.join(frames_folder, f)
        img = cv2.imread(path)
        if img is None:
            continue
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.3, 5)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(cropped_folder, f), face)

# ------------------------------------------------------------
# Step 4: Resize (64x64 RGB) + Normalize
# ------------------------------------------------------------
def resize_and_normalize(cropped_folder, resized_folder, normalized_folder, size=(64, 64)):
    ensure_dir(resized_folder)
    ensure_dir(normalized_folder)

    for f in os.listdir(cropped_folder):
        if not f.lower().endswith(".jpg"):
            continue
        path = os.path.join(cropped_folder, f)
        img = cv2.imread(path)
        if img is None:
            continue
        resized = cv2.resize(img, size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        cv2.imwrite(os.path.join(resized_folder, f), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(os.path.join(normalized_folder, f.replace(".jpg", ".npy")), norm)

# ------------------------------------------------------------
# Step 5: Augment Images
# ------------------------------------------------------------
def augment_images(resized_folder, aug_folder, augment_count=5):
    ensure_dir(aug_folder)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest"
    )

    for f in os.listdir(resized_folder):
        if not f.lower().endswith(".jpg"):
            continue
        path = os.path.join(resized_folder, f)
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(img_rgb, axis=0)
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            aug_img = batch[0].astype(np.uint8)
            out_name = f.replace(".jpg", f"_aug_{i}.jpg")
            cv2.imwrite(os.path.join(aug_folder, out_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            i += 1
            if i >= augment_count:
                break

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def run_pipeline(root="Face_Dataset/Videos", n_frames=120):
    videos = list_videos_recursive(root)
    if not videos:
        print(f" No videos found in {root}")
        return

    grouped = defaultdict(list)
    for path, person in videos:
        grouped[person].append(path)

    print(f"Found {len(grouped)} persons: {list(grouped.keys())}")

    for person, vids in grouped.items():
        print(f"\nProcessing {person} ...")

        frames_folder = f"Face_Dataset/Frames/{person}"
        grey_folder = f"Face_Dataset/Grey/{person}"
        binary_folder = f"Face_Dataset/Binary/{person}"
        cropped_folder = f"Face_Dataset/Cropping/{person}"
        resized_folder = f"Face_Dataset/Resized_rgb(64X64)/{person}"
        normalized_folder = f"Face_Dataset/Normalized/{person}"
        aug_folder = f"Face_Dataset/Augmented/{person}"

        # Create dirs
        for p in [frames_folder, grey_folder, binary_folder, cropped_folder,
                  resized_folder, normalized_folder, aug_folder]:
            ensure_dir(p)

        # Extract frames
        for v in vids:
            count = extract_n_frames(v, frames_folder, n_frames)
            print(f" {os.path.basename(v)} → {count} frames")

        # Convert Greyscale + Binary
        convert_to_greyscale_and_binary(frames_folder, grey_folder, binary_folder)
        print("Converted to Grey & Binary")

        # Crop Faces
        crop_faces(frames_folder, cropped_folder)
        print(" Cropped faces")

        # Resize & Normalize
        resize_and_normalize(cropped_folder, resized_folder, normalized_folder, size=(64, 64))
        print("Resized (64x64) & Normalized")

        # Augment
        augment_images(resized_folder, aug_folder, augment_count=5)
        print("Augmented images")

    print("\nPipeline Completed for All Persons!")

# ------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(root="Face_Dataset/Videos", n_frames=120)
