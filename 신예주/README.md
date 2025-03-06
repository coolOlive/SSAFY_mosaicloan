### 목차
1. [250304_화](#250304_화)
2. [250305_수](#250305_수)
3. [250306_목](#250306_목)

# 250304_화

## 1. 크롤링을 통해 수집한 뉴스 데이터 감정분석
### 모델  
#### KLUE BERT base
  
- 한국어 자연어 처리를 위해 개발된 언어 모델  
- 벤치마크 데이터셋과 모델 제공
영문 데이터셋인 Finance Phrase Bank를 한국어로 번역하고 육안 검수한 데이터셋으로 pre-trained된 모델 사용 예정
- 감정 라벨 : 긍정 / 부정 / 중립  

성능 및 활용
- 문서 분류(Document Classification)
- 개체명 인식(Named Entity Recognition)
- 관계 추출(Relation Extraction)
- 의미역 결정(Semantic Role Labeling)
- 감정 분석(Sentiment Analysis)
- 자연어 추론(Natural Language Inference)
- 의도 분류(Intent Classification)  

장단점
- 장점: BERT 계열 모델 중 한국어 처리에 최적화
- 단점: base 모델은 large 모델 보다 파라미터 수가 적어 일부 복잡한 테스크에 서는 성능 차이가 있을 수 있음 


참고
https://www.blog.data101.io/394

## 2. 신용등급평가를 위한 데이터셋
### Home Credit Default Loan

금융 기관의 신용 평가 및 대출 관련 데이터

#### 주요 변수 카테고리

1. 고객 인적 정보

    - 고객의 성별, 생년월일, 교육 수준, 결혼 상태, 자녀 수, 직업 등
    - 주소 정보 (지역, 우편번호)
    - 고용 정보 (고용 시작일, 고용주, 업종)

2. 신용 계약 정보

    - 대출 금액, 금리, 월 상환액, 상환 기간
    - 계약 시작일, 종료일, 활성화 날짜
    - 담보 유형 및 가치
    - 대출 목적

3. 상환 이력
    - 연체일(DPD: Days Past Due)
    - 최대 연체일 수
    - 납부 완료된 할부금 수
    - 미납 할부금 수
    - 정시 납부 및 조기 납부 이력

4. 신용 조회 정보
    - 신용 조회 횟수 (30일, 90일, 120일, 180일, 360일 기간별)
    - 마지막 신용 조회 날짜
    - 신용 등급/평가

5. 기존 대출 정보
    - 현재 활성 대출 수
    - 총 부채 금액
    - 미납 부채 금액
    - 신용 한도

6. 신청 처리 정보
    - 이전 신청 상태 (승인, 거절)
    - 거절 사유
    - 취소 사유
    - 신청 일자

7. 거래 정보
    - 입출금 거래 금액
    - 카드 거래 내역
    - 예금 잔액

8. 세금 공제 정보
    - 세금 공제 날짜
    - 세금 공제 금액  

주의사항
- 데이터를 원하는 형태로 변환 후 사용해야됨  

참고

https://www.kaggle.com/code/sagarmamodia/home-credit-default-loan-notebook

---
### HMEQ Data

주택담보대출 관련 데이터

#### 특징
- target  
    - 0 = 대출 상환, 1 = 대출 연체 또는 채무불이행
- 파일 형식
    - csv 파일로 되어있음

참고  
https://www.kaggle.com/datasets/ajay1735/hmeq-data  
</br>
</br>

# 250305_수
## 1. 데이터셋
### Home Credit Default Loan
csv 데이터 수집 및 정리를 통해 시각화를 진행하고자 했으나 데이터의 용량이 커서 다른 데이터를 이용하기로 함
### HMEQ 
데이터의 특성을 파악하기 위한 간단한 시각화를 진행하였으나, 원하는 정보를 담고 있지 않아 새로운 데이터를 탐색하기로 함
### Loan Classification Dataset  
대출 가능 여부를 파악하기 위한 데이터로 한번 더 검토가 필요함  

## 2. 피그마
피그마 피드백을 통해 프로젝트 프레임 워크를 다함께 정함
<br><br>

# 250306_목  

## 1. 데이터셋
### Loan Classification Dataset  
데이터셋 검토 후 전처리 및 결측치 처리에 대해 생각하는 시간을 가짐
- 데이터 전처리 전략
    - 상관관계가 높은 변수들을 통합하거나 대표 변수 선택
    - 비슷한 성격의 변수들을 새로운 복합 지표로 생성
    - null 값이 많은 열 처리 방법(dropna, fillna 등)

- 대출 관련 용어
    - RejectStats: 대출 거부 통계, 거부된 대출 신청에 관한 데이터
    - FICO 점수: 미국에서 사용되는 신용 평가 시스템(300-850점)
    - Charge-off: 대출 기관이 더 이상 상환을 받을 수 없다고 판단하여 손실로 처리하는 것     
                                                             
### Home Credit Default Loan
마지막에 결국 해당 데이터셋을 사용한 모델을 만들고자 하며, Hadoop에서 Spark를 이용하기 위해 환경 셋팅을 진행함.
- Java JDK 설치
- Apache Spark 설치
- PySpark 및 Jupyter 패키지 설치
- Spark와 Jupyter 연동 테스트