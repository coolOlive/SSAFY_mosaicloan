### 목차
1. [250304_화](#250304_화)

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
https://www.kaggle.com/code/sagarmamodia/home-credit-default-loan-notebook/input
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

