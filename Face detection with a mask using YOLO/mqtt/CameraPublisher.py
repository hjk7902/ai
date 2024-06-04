import cv2
import paho.mqtt.client as mqtt
import threading
import base64


class ImageMqttPublisher:
    def __init__(self, brokerIp=None, brokerPort=1883, pubTopic=None):
        self.brokerIp = brokerIp
        self.brokerPort = brokerPort
        self.pubTopic = pubTopic
        self.client = None

    def connect(self):
        thread = threading.Thread(target=self.__run, daemon=True)
        thread.start()

    def __run(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.__on_connect
        self.client.on_disconnect = self.__on_disconnect
        self.client.connect(self.brokerIp, self.brokerPort)
        self.client.loop_forever()

    def __on_connect(self, client, userdata, flags, rc):
        print("ImageMqttClient mqtt broker connected")

    def __on_disconnect(self, client, userdata, rc):
        print("ImageMqttClient mqtt broker disconnected")

    def disconnect(self):
        self.client.disconnect()

    def sendBase64(self, frame):
        if self.client is None:
            return
        # MQTT Broker가 연결되어 있지 않을 경우
        if not self.client.is_connected():
            return
        # JPEG 포맷으로 인코딩
        retval, bytes = cv2.imencode(".jpg", frame)
        # 인코딩이 실패났을 경우
        if not retval:
            print("image encoding fail")
            return
        # Base64 문자열로 인코딩
        b64_bytes = base64.b64encode(bytes)
        # MQTT Broker에 보내기
        self.client.publish(self.pubTopic, b64_bytes, retain=True)


if __name__ == "__main__":
    videoCapture = cv2.VideoCapture("rtsp://username:password@hostname:port/profile")

    imageMqttPusblisher = ImageMqttPublisher("localhost", 1883, "/camerapub")
    imageMqttPusblisher.connect()
    t = 0
    while True:
        if videoCapture.isOpened():
            retval, frame = videoCapture.read()
            frame = cv2.resize(frame, (640,480))
            if not retval:
                print("video capture fail")
                break
            imageMqttPusblisher.sendBase64(frame)
            print("\r", t, end="")
            t += 1
        else:
            break

    imageMqttPusblisher.disconnect()
    videoCapture.release()