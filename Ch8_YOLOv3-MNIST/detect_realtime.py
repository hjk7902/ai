import cv2
import numpy as np
import tensorflow as tf

from image_process import resize_to_square
from data import read_class_names
from post_process import *
from yolov3 import Create_YOLOv3
from nms import nms

# -------------------------
# Model
# -------------------------
yolo = Create_YOLOv3(
    num_class=10,
    input_shape=(416, 416, 3),
    train_mode=False
)
yolo.load_weights("checkpoints/mnist_custom_best.weights.h5")

class_names = read_class_names("dataset/mnist.names")

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ 연결된 카메라가 없습니다.")
    exit()

print("🎥 Camera started")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 받지 못했습니다.")
        break

    # 밝기를 100만큼 더함 
    dummy = np.full(frame.shape, fill_value=100, 
                    dtype=np.uint8)
    cv2.add(frame, dummy, frame)
            
    # 콘트라스트 강조함 
    image = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    
    # resize (letterbox)
    image_data = resize_to_square(np.copy(frame), 416)
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # forward
    pred_bbox = yolo(image_data, training=False)

    # flatten
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0).numpy()

    # post-process
    bboxes = postprocess_boxes(
        pred_bbox,
        frame,
        input_size=416,
        score_threshold=0.5
    )

    bboxes = nms(bboxes, iou_threshold=0.3)

    # draw
    result = draw_bbox(frame, bboxes, class_names)

    cv2.imshow("YOLOv3 Realtime Detection", result)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("🎥 Camera released")