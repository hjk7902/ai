import numpy as np
from bbox_iou import bbox_iou

def nms(bboxes, iou_threshold=0.5, sigma=0.3, method='nms', score_threshold=0.001):
    if len(bboxes) == 0:
        return []

    classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # 1. 경계 상자의 개수가 0보다 큰지 확인
        while len(cls_bboxes) > 0:
            # 2. 점수 순서 A에 따라 가장 높은 점수를 갖는 경계 상자를 선택
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # 3. 경계 상자 A를 계산하고 경계 상자의 모든 iou를 계산하고 iou 값이 임계값보다 높은 경계 상자를 제거
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4], method='iou', isTrain=False)
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold # score_threshold
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


import matplotlib.pyplot as plt

def visualize_nms(true_boxes, pred_boxes, iou_threshold=0.5,
    sigma=0.3, method='nms', score_threshold=0.001):

    def draw(ax, boxes, color, ls, label):
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b[:4]
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linestyle=ls,
                linewidth=2, label=label if i == 0 else None
            ))
            if b.shape[0] > 4:
                ax.text(x1, y1 - 2, f"{b[4]:.2f}", color=color, fontsize=20, va='top')

    nms_boxes = nms(
        pred_boxes.copy(),
        iou_threshold=iou_threshold,
        sigma=sigma,
        method=method,
        score_threshold=score_threshold
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, boxes, title in zip(axes,
        [pred_boxes, np.array(nms_boxes)],
        ['Before NMS', 'After NMS']
    ):
        draw(ax, true_boxes, 'red', '-', 'GT')
        draw(ax, boxes, 'blue', '--', 'Pred')
        ax.set(title=f"{title} ({method})", xlim=(0, 220), ylim=(0, 220))
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    true_boxes = np.array([[50, 50, 150, 150]])
    pred_boxes = np.array([
        [48, 48, 152, 152, 0.95, 0],
        [60, 60, 160, 160, 0.80, 0],
        [40, 40, 140, 140, 0.70, 0],
        [120, 120, 200, 200, 0.60, 0],
    ])

    visualize_nms(true_boxes, pred_boxes, method='nms')
    visualize_nms(true_boxes, pred_boxes, method='soft-nms', sigma=0.5)
    visualize_nms(true_boxes, pred_boxes, method='soft-nms',
        sigma=0.5, score_threshold=0.4
    )