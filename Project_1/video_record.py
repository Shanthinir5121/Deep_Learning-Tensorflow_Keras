import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time


# RECORD ONE VIDEO

def record_video(person_name, lighting_condition):
    base_dir = f"Face_Dataset/Videos/{person_name}"
    os.makedirs(base_dir, exist_ok=True)
    filename = f"{base_dir}/{lighting_condition}.avi"

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    print(f"\n🎥 Recording video for {person_name} under {lighting_condition} (1 minute max)")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped manually.")
            break
        if time.time() - start_time > 60:
            print("Recording completed (1 minute).")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved video: {filename}")
    return filename