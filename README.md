# 인공신경망 딥러닝 알고리즘 구현과 실시간 객체 탐지
<img width="642" height="512" alt="20251214_040715" src="https://github.com/user-attachments/assets/2b8fa49a-2200-4bff-98fb-c04023434f0a" />

## 모델 다운로드
https://drive.google.com/file/d/1vckKHij7dZPBit3NkvCFfrWVFTMuLPpf/view?usp=sharing

## 교재 주문
* https://bookk.co.kr/bookStore/646af1334222b24502d478a6

## tensowflow 2.20 버전
### 라이브러리 설치
* pip install -r requirements.txt

  
## 교재 수정 내용
### 2025. 12. 23.
 - 텐서플로우 2.20 호환 코드로 전면 교채
 - 교재 개정판 출간 예정(2026년 1월경)

### 2024. 7. 1.
 - 교재의 javaspecialist.co.kr에 있는 자료를 모두 깃허브로 옮김
 - https://javaspeicalist.co.kr -> htts://github.com/hjk7902/ai
 - YOLOv3 모델의 클래스 수가 80개일 경우 총 파라미터 개수 수정( 61,624,807-> 62,001,757)
 - 8장의 모델을 내려받는 방법을 수정했습니다.(단축주소 -> 깃허브의 README 파일 링크)
 - 변수명 수정: train_X -> X_train, test_X -> X_test, train_y -> y_train, test_y -> y_test
   
### 2023. 10. 20.
 - 1장 2절의 인공신경망 딥러닝 모델 구현에 사용한 데이터의 종속변수를 sklearm과 keras를 이용한 모델에서 원-핫 인코딩해서 사용하지 않음
 - 텐서플로우의 코드만 원-핫 인코딩을 사용하도록 했음 
 - GoogLeNet 인셉션 모듈 코드 예 추

### 2023. 8. 25.
 - 6장 4.5절에 YOLO 버전 별 출시 시점과 특징 추가 및 일부 내용 수정 

### 2023. 6. 14.
 - YOLOv8, YOLO 리뷰 참고문헌추가

### 2023. 6. 4.
 - p.119, 14라인 np.argmax(test_y[i])를 test_y[i]로 수정

### 2022. 12. 1.
 - 참고문헌 DOI 수정
 - YOLOv4,6,7 참고문헌 추가

### 2022. 10. 1.
 - 쌍점(:) 앞에 공백 없앰
 - 5장 마지막 빈 페이지에 인공신경망 요약정리 그림 삽입

----------------------------------------------------------------------------------
## tensowflow 2.14 이하 버전

### 1. 가상환경 추가 
* conda create -n tf2.14 python=3.11
* conda activate tf2.14

### 2. 주피터노트북 ipykernel 추가 
* pip install ipykernel
* ipython kernel install --name=tf2.14 --user

### 3. 라이브러리 설치
* python -m pip install pip==23.0   <- pip 버전을 다운그래이드 해야 텐서플로우 2.14 설치 가능
* pip install tensorflow==2.14 
* pip install opencv-python scikit-learn pandas nltk ipywidgets tqdm datasets
* pip install "numpy<2.0" --user

### 4. 학습된 모델
* YOLOv3 MNIST 모델(tf 2.14이하 버전)
[checkpoints_mnist.zip](https://drive.google.com/file/d/19udN0Q881hFrYQ-eEuXQjywFXoRFzKl0/view?usp=sharing)

* YOLOv3 Mask 모델(tf 2.14이하 버전)
[checkpoints_mask.zip](https://drive.google.com/file/d/16Lzowa8Hh4ggCcGCBC1Qgf89VEI8o0WU/view?usp=sharing)

