# 인공신경망 딥러닝 알고리즘 구현과 실시간 객체 탐지

## 교재 주문
* https://bookk.co.kr/bookStore/646af1334222b24502d478a6

## 학습된 모델
### YOLOv3 MNIST 모델
https://drive.google.com/file/d/12c0Ke8cxJ1zx4cCzuGbQHq2UCBwS-CPb/view?usp=sharing

### YOLOv3 Mask 모델
https://drive.google.com/file/d/14MakQl0__okb8ivNp0mn9rRPdBAnbWTi/view?usp=sharing

## 교재 수정 내용
### 2024. 7. 1.
 - 교재의 javaspecialist.co.kr에 있는 자료를 모두 깃허브로 옮김
 - https://javaspeicalist.co.kr -> htts://github.com/hjk7902/ai
 - YOLOv3 모델의 클래스 수가 80개일 경우 총 파라미터 개수 수정( 61,624,807-> 62,001,757)
 - 8장의 모델을 내려받는 방법을 수정했습니다.(단축주소 -> 깃허브의 README 파일 링크)
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
