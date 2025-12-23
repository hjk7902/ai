import numpy as np
import cv2
import tensorflow as tf

from config import NUM_CLASS
from image_process import resize_to_square
from data import read_class_names
from post_process import *
from yolov3 import Create_YOLOv3
from nms import nms


def detect_image(model, image_path, output_path,
                 class_label_path,
                 input_size=416, show=False,
                 score_threshold=0.3, iou_threshold=0.45,
                 rectangle_colors=''):

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Image not found: {image_path}")

    class_names = read_class_names(class_label_path)

    # 1️⃣ letterbox resize
    image_data = resize_to_square(
        np.copy(original_image),
        target_size=input_size
    )
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # 2️⃣ forward
    pred_bbox = model(image_data, training=False)

    # 3️⃣ flatten
    pred_bbox = [
        tf.reshape(p, (-1, 5 + NUM_CLASS))
        for p in pred_bbox
    ]
    pred_bbox = tf.concat(pred_bbox, axis=0).numpy()
    print("pred_bbox shape:", pred_bbox.shape)
    print(np.unique(pred_bbox[:, -1].astype(int)))
    print("pred_bbox sample:", pred_bbox[:5])

    # 4️⃣ post-process
    bboxes = postprocess_boxes(
        pred_bbox,
        original_image,
        input_size,
        score_threshold
    )
    print("after postprocess:", len(bboxes))

    if len(bboxes) == 0:
        print("⚠️ No objects detected")

    bboxes = nms(bboxes, iou_threshold)
    print(bboxes[:5])

    # 5️⃣ draw
    image = draw_bbox(
        original_image,
        bboxes,
        class_names,
        rectangle_colors=rectangle_colors
    )

    if output_path:
        cv2.imwrite(output_path, image)

    if show:
        cv2.imshow("predicted image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def main():
    yolo = Create_YOLOv3(
        num_class=NUM_CLASS,
        input_shape=(416, 416, 3),
        train_mode=False   # 반드시 False
    )

    yolo.load_weights("checkpoints/mnist_custom_best.weights.h5")

    detect_image(
        model=yolo,
        image_path="mnist_test_c.jpg",
        output_path="result.jpg",
        class_label_path="dataset/mnist.names",
        show=True,
        score_threshold=0.65,
        iou_threshold=0.3
    )

if __name__ == "__main__":
    main()