import os
import shutil
import tensorflow as tf
from data import DataGenerator
from config import *
from bbox_iou import bbox_iou
from yolov3 import Create_YOLOv3

if os.path.exists(LOGDIR): 
    shutil.rmtree(LOGDIR)

writer = tf.summary.create_file_writer(LOGDIR)
validate_writer = tf.summary.create_file_writer(LOGDIR)


def compute_loss(pred, conv, label, bboxes, i=0, iou_loss_thresh=IOU_THRESHOLD):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_iou(pred_xywh, label_xywh, method=IOU_METHOD), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    iou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    # iou_xywh
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :], method=IOU_METHOD)
    # 실제 상자에서 가장 큰 예측값을 갖는 상자로 IoU 값 찾기
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # 가장 큰 iou가 임계값보다 작으면 예측 상자에 개체가 포함되지 않은 것으로 간주되고 배경 상자로 설정
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < iou_loss_thresh, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Confidence의 loss 계산
    # 그리드에 객체가 포함된 경우 1, 그렇지 않을경우 0
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return iou_loss, conf_loss, prob_loss


def train_step(model, image_data, target, lr_init=1e-4, lr_end=1e-6):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)

        iou_loss = conf_loss = prob_loss = 0.0

        for i in range(3):
            conv = pred_result[i * 2]       # raw conv
            pred = pred_result[i * 2 + 1]   # decoded pred

            loss_items = compute_loss(pred, conv, *target[i], i)

            iou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = iou_loss + conf_loss + prob_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # learning rate update (당신 코드 그대로)
    global_steps.assign_add(1)
    if global_steps < warmup_steps:
        lr = global_steps / warmup_steps * lr_init
    else:
        lr = lr_end + 0.5 * (lr_init - lr_end) * (
            1 + tf.cos((global_steps - warmup_steps) /
                       (total_steps - warmup_steps) * np.pi)
        )

    optimizer.learning_rate.assign(lr)

    return (
        global_steps.numpy(),
        optimizer.learning_rate.numpy(),
        iou_loss.numpy(),
        conf_loss.numpy(),
        prob_loss.numpy(),
        total_loss.numpy(),
    )


def validate_step(model, image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=False)
        iou_loss = conf_loss = prob_loss = 0 

        grid = 3 
        for i in range(grid):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            iou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = iou_loss + conf_loss + prob_loss

    return iou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()


trainset = DataGenerator(data_path="dataset/mnist_train",
                         annot_path="dataset/mnist_train.txt",
                         class_label_path="dataset/mnist.names")
testset = DataGenerator(data_path="dataset/mnist_test", 
                        annot_path="dataset/mnist_test.txt",
                        class_label_path="dataset/mnist.names")
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64) 
warmup_steps = WARMUP_EPOCHS * steps_per_epoch
total_steps = EPOCHS * steps_per_epoch

optimizer = tf.keras.optimizers.Adam()


def main():
    yolo = Create_YOLOv3(num_class=NUM_CLASS, train_mode=True)

    best_val_loss = float("inf")
    os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)

    for epoch in range(EPOCHS):
        for step, (image_data, target) in enumerate(trainset):
            results = train_step(yolo, image_data, target)
            cur_step = step + 1

            print(
                f"epoch:{epoch:3d} "
                f"step:{cur_step:5d}/{steps_per_epoch}, "
                f"lr:{results[1]:.6f}, "
                f"iou_loss:{results[2]:7.2f}, "
                f"conf_loss:{results[3]:7.2f}, "
                f"prob_loss:{results[4]:7.2f}, "
                f"total_loss:{results[5]:7.2f}", end='\r'
            )

        if len(testset) == 0:
            print("No validation set. Saving last model.")

            last_path = os.path.join(
                CHECKPOINTS_FOLDER,
                f"{MODEL_NAME}_epoch_{epoch:03d}{MODEL_EXTENSION}"
            )
            yolo.save_weights(last_path)
            continue

        count = 0
        iou_val = conf_val = prob_val = total_val = 0.0

        for image_data, target in testset:
            results = validate_step(yolo, image_data, target)
            count += 1
            iou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]

        iou_val /= count
        conf_val /= count
        prob_val /= count
        total_val /= count

        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total", total_val, step=epoch)
            tf.summary.scalar("validate_loss/iou", iou_val, step=epoch)
            tf.summary.scalar("validate_loss/conf", conf_val, step=epoch)
            tf.summary.scalar("validate_loss/prob", prob_val, step=epoch)
        validate_writer.flush()

        print(
            f"\n[Epoch {epoch:03d}] "
            f"iou:{iou_val:.2f}, conf:{conf_val:.2f}, "
            f"prob:{prob_val:.2f}, total:{total_val:.2f}"
        )

        if SAVE_CHECKPOINT:
            epoch_path = os.path.join(
                CHECKPOINTS_FOLDER,
                f"{MODEL_NAME}_epoch_{epoch:03d}_val_{total_val:.2f}{MODEL_EXTENSION}"
            )
            yolo.save_weights(epoch_path)

        if SAVE_BEST_ONLY and total_val < best_val_loss:
            best_val_loss = total_val
            best_path = os.path.join(
                CHECKPOINTS_FOLDER,
                f"{MODEL_NAME}_best{MODEL_EXTENSION}"
            )
            yolo.save_weights(best_path)

            print(f"Best model updated (val_loss={best_val_loss:.2f})")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs:", gpus)

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()