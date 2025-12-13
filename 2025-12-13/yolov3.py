import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# ---------------------------
# 하이퍼/상수
# ---------------------------
YOLO_STRIDES  = [8, 16, 32]
YOLO_ANCHORS  = [[[10,  13], [16,   30], [33,   23]],
                 [[30,  61], [62,   45], [59,  119]],
                 [[116, 90], [156, 198], [373, 326]]]
STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T / STRIDES).T   # 기존 방식 유지

# 텐서 상수로도 준비
TF_STRIDES = tf.constant(STRIDES, dtype=tf.float32)
TF_ANCHORS = tf.constant(ANCHORS, dtype=tf.float32)  # shape (3,3,2)

# ---------------------------
# BatchNormalization subclass
# ---------------------------
class BatchNormalization(layers.BatchNormalization):
    # "동결 상태(Frozen state)"와 "추론 모드(Inference mode)"는 별개의 개념입니다. 
    # 'layer.trainable=False' 이면 레이어를 동결시킵니다. 이것은 훈련하는 동안 내부 상태 즉, 가중치가 바뀌지 않습니다.
    # 그런데 layer.trainable=False이면 추론 모드로 실행됩니다. 
    # 레이어는 추론모드에서 현재 배치의 평균 및 분산을 사용하는 대신 현재 배치를 정규화하기 위해 이동 평균과 이동 분산을 사용합니다.
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

# ---------------------------
# convolutional / residual / darknet53 
# ---------------------------
def convolutional(input_layer, filters, kernel_size,
                  downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    kernel_init = tf.random_normal_initializer(stddev=0.01)
    conv = layers.Conv2D(filters=filters, 
                         kernel_size=kernel_size,
                         strides=strides, padding=padding, 
                         use_bias=not bn,
                         kernel_initializer=kernel_init,
                         kernel_regularizer=l2(0.0005)
                        )(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate:
        conv = layers.LeakyReLU(alpha=0.1)(conv)

    return conv


def residual_block(input_layer, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters=filter_num1, kernel_size=(1,1))
    conv = convolutional(conv       , filters=filter_num2, kernel_size=(3,3))
    residual_output = short_cut + conv
    return residual_output


def darknet53(input_data):
    input_data = convolutional(input_data, 32, (3,3))
    input_data = convolutional(input_data, 64, (3,3), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  32, 64)

    input_data = convolutional(input_data, 128, (3,3), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 64, 128)

    input_data = convolutional(input_data, 256, (3,3), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, 512, (3,3), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, 1024, (3,3), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 512, 1024)

    return route_1, route_2, input_data

# ---------------------------
# Upsample Layer
# ---------------------------
class Upsample(layers.Layer):
    def __init__(self, method='nearest', **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def call(self, x):
        # x.shape[1], x.shape[2]는 KerasTensor일 수 있으므로 동적 표현 사용
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        new_size = (h * 2, w * 2)
        return tf.image.resize(x, new_size, method=self.method)

def upsample(input_layer):
    # 이전 함수명이랑 충돌을 방지하기 위해 functional-level wrapper를 제공
    return Upsample()(input_layer)


# ---------------------------
# YOLOv3 네트워크 정의
# ---------------------------
def YOLOv3(input_layer, num_class):
    route_1, route_2, conv = darknet53(input_layer)

    conv = convolutional(conv, 512, (1, 1))
    conv = convolutional(conv, 1024, (3, 3))
    conv = convolutional(conv, 512, (1, 1))
    conv = convolutional(conv, 1024, (3, 3))
    conv = convolutional(conv, 512, (1, 1))
    conv_lobj_branch = convolutional(conv, 1024, (3, 3))

    conv_lbbox = convolutional(conv_lobj_branch,
                               3 * (num_class + 5), (1, 1),
                               activate=False, bn=False)

    conv = convolutional(conv, 256, (1, 1))
    conv = Upsample()(conv)

    # 🔥 tf.concat → Keras Concatenate
    conv = layers.Concatenate(axis=-1)([conv, route_2])

    conv = convolutional(conv, 256, (1, 1))
    conv = convolutional(conv, 512, (3, 3))
    conv = convolutional(conv, 256, (1, 1))
    conv = convolutional(conv, 512, (3, 3))
    conv = convolutional(conv, 256, (1, 1))
    conv_mobj_branch = convolutional(conv, 512, (3, 3))

    conv_mbbox = convolutional(conv_mobj_branch,
                               3 * (num_class + 5), (1, 1),
                               activate=False, bn=False)

    conv = convolutional(conv, 128, (1, 1))
    conv = Upsample()(conv)

    # 🔥 tf.concat → Keras Concatenate
    conv = layers.Concatenate(axis=-1)([conv, route_1])

    conv = convolutional(conv, 128, (1, 1))
    conv = convolutional(conv, 256, (3, 3))
    conv = convolutional(conv, 128, (1, 1))
    conv = convolutional(conv, 256, (3, 3))
    conv = convolutional(conv, 128, (1, 1))
    conv_sobj_branch = convolutional(conv, 256, (3, 3))

    conv_sbbox = convolutional(conv_sobj_branch,
                               3 * (num_class + 5), (1, 1),
                               activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


# ---------------------------
# DecodeLayer (모든 TF 연산을 Layer 내부에서 처리)
# ---------------------------
class DecodeLayer(layers.Layer):
    def __init__(self, num_class, scale_idx, **kwargs):
        super().__init__(**kwargs)
        self.num_class = num_class
        self.scale_idx = scale_idx
        self.stride = STRIDES[scale_idx]
        self.anchors = ANCHORS[scale_idx]

    def call(self, conv):
        """
        conv: (B, H, W, A*(5+num_class))
        return: (B, H, W, A, 5+num_class)
        """
        batch_size = tf.shape(conv)[0]
        output_size = tf.shape(conv)[1]
        anchor_per_scale = len(self.anchors)

        # (B, H, W, A, 5+num_class)
        conv = tf.reshape(
            conv,
            (batch_size,
             output_size,
             output_size,
             anchor_per_scale,
             5 + self.num_class)
        )

        # 분해
        conv_raw_dxdy = conv[..., 0:2]
        conv_raw_dwdh = conv[..., 2:4]
        conv_raw_conf = conv[..., 4:5]
        conv_raw_prob = conv[..., 5:]

        # grid 생성
        grid_y = tf.range(output_size, dtype=tf.int32)
        grid_x = tf.range(output_size, dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        grid = tf.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)
        grid = tf.expand_dims(grid, axis=2)         # (H, W, 1, 2)
        grid = tf.cast(grid, tf.float32)

        # xy decode
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + grid) * self.stride

        # wh decode
        pred_wh = (tf.exp(conv_raw_dwdh) *
                   tf.cast(self.anchors, tf.float32)) * self.stride

        # confidence & class prob
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        # 최종 출력
        return tf.concat(
            [pred_xy, pred_wh, pred_conf, pred_prob],
            axis=-1
        )


# ---------------------------
# 모델 생성 헬퍼
# ---------------------------
def Create_YOLOv3(num_class, input_shape=(416,416,3), train_mode=False):
    input_layer  = layers.Input(input_shape)
    conv_tensors = YOLOv3(input_layer, num_class)  # raw conv outputs (3 tensors)

    decoded_tensors = []
    for i, conv in enumerate(conv_tensors):
        decoded = DecodeLayer(num_class, i)(conv)
        decoded_tensors.append(decoded)

    outputs = []
    if train_mode:
        # training: raw conv tensors + decoded tensors (training loss가 raw conv tensor 기준이라면 유연하게 사용)
        for conv, dec in zip(conv_tensors, decoded_tensors):
            outputs.append(conv)   # raw conv
            outputs.append(dec)    # decoded
    else:
        # inference: decoded tensors만
        outputs = decoded_tensors

    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    return model

# ---------------------------
# 사용 예시
# ---------------------------
if __name__ == "__main__":
    NUM_CLASS = 10
    model = Create_YOLOv3(num_class=NUM_CLASS, input_shape=(416,416,3), train_mode=True)
    model.summary()
