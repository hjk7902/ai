import numpy as np
import tensorflow as tf

def bbox_iou(
    boxes1,
    boxes2,
    method="iou",      # "iou", "giou", "diou", "ciou", "ciou_v5"
    isTrain=True,
    eps=1e-8
):
    """
    boxes1, boxes2
      - isTrain=True  : (x, y, w, h)  [Tensor]
      - isTrain=False : (xmin, ymin, xmax, ymax) [ndarray]
    """

    # =========================
    # 1. 평가용 (NMS)
    # =========================
    if not isTrain:
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter = np.maximum(right_down - left_up, 0.0)
        inter_area = inter[..., 0] * inter[..., 1]
        union = area1 + area2 - inter_area
        union = np.maximum(union, eps)

        return np.maximum(inter_area / union, np.finfo(np.float32).eps)

    # =========================
    # 2. xywh → xyxy
    # =========================
    b1_xyxy = tf.concat(
        [boxes1[..., :2] - boxes1[..., 2:] * 0.5,
         boxes1[..., :2] + boxes1[..., 2:] * 0.5],
        axis=-1
    )
    b2_xyxy = tf.concat(
        [boxes2[..., :2] - boxes2[..., 2:] * 0.5,
         boxes2[..., :2] + boxes2[..., 2:] * 0.5],
        axis=-1
    )

    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]

    left_up = tf.maximum(b1_xyxy[..., :2], b2_xyxy[..., :2])
    right_down = tf.minimum(b1_xyxy[..., 2:], b2_xyxy[..., 2:])

    inter = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter[..., 0] * inter[..., 1]
    union = area1 + area2 - inter_area
    union = tf.maximum(union, eps)

    iou = inter_area / union

    if method == "iou":
        return iou

    # =========================
    # 3. enclosing box (공통)
    # =========================
    enclose_left = tf.minimum(b1_xyxy[..., :2], b2_xyxy[..., :2])
    enclose_right = tf.maximum(b1_xyxy[..., 2:], b2_xyxy[..., 2:])
    enclose = tf.maximum(enclose_right - enclose_left, 0.0)

    c2 = tf.reduce_sum(tf.square(enclose), axis=-1) + eps

    center_dist = tf.reduce_sum(
        tf.square(boxes1[..., :2] - boxes2[..., :2]), axis=-1
    )

    # =========================
    # 4. GIoU
    # =========================
    if method == "giou":
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - (enclose_area - union) / (enclose_area + eps)
        return giou

    # =========================
    # 5. DIoU
    # =========================
    if method == "diou":
        return iou - center_dist / c2

    # =========================
    # 6. CIoU (논문 원본)
    # =========================
    if method == "ciou":
        ar_gt = boxes2[..., 2] / (boxes2[..., 3] + eps)
        ar_pred = boxes1[..., 2] / (boxes1[..., 3] + eps)

        v = 4 / (np.pi ** 2) * tf.square(tf.atan(ar_gt) - tf.atan(ar_pred))
        alpha = v / (1 - iou + v + eps)

        return iou - (center_dist / c2 + alpha * v)

    # =========================
    # 7. CIoU (YOLOv5 / YOLOv8)
    # =========================
    if method == "ciou_v5":
        ar_gt = boxes2[..., 2] / (boxes2[..., 3] + eps)
        ar_pred = boxes1[..., 2] / (boxes1[..., 3] + eps)

        v = 4 / (np.pi ** 2) * tf.square(tf.atan(ar_gt) - tf.atan(ar_pred))

        alpha = tf.stop_gradient(
            v / (1 - iou + v + eps)
        )

        return iou - (center_dist / c2 + alpha * v)

    raise ValueError(f"Unknown method: {method}")


import matplotlib.pyplot as plt

def visualize_iou(true, pred, ious=None):
    n = len(true)
    fig, axes = plt.subplots(1, n, figsize=(8, 5))
    axes = [axes] if n == 1 else axes

    for i, ax in enumerate(axes):
        for box, color, style, label in (
            (true[i], "red", "-", "True"),
            (pred[i], "blue", "--", "Pred")
        ):
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color,
                linestyle=style, linewidth=2
            ))
            ax.text(x1, y1-5, label, color=color, fontsize=10)

        ax.set_title(f"IoU = {ious[i]:.4f}" if ious is not None else f"Case {i}")
        ax.set(xlim=(0, 220), ylim=(220, 0), aspect="equal")
        ax.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # 테스트 코드
    gt_boxes = np.array([[50, 50, 150, 150], [30, 30, 120, 120]]) # True boxes
    pred_boxes = np.array([[100, 100, 200, 200], [50, 50, 150, 150]]) # Predicted boxes

    ious = bbox_iou(gt_boxes, pred_boxes, isTrain=False)
    print("IoUs:", ious)

    visualize_iou(gt_boxes, pred_boxes, ious)