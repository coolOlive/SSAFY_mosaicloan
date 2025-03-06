### 목차
1. [250304_화](#250304_화)
1. [250305_수](#250305_수수)
---
# 250304_화

## 뉴스 기사 감성 분석 모델 학습

1. 경제 관련 뉴스 기사를 크롤링하여 만들어진 빅데이터를 분석하여 경제 시장 분위기의 긍정 /  부정 / 중립 여부를 평가하고자 했다.

2. 크롤링한 데이터는 긍정 / 부정 / 중립 여부가 라벨링 되어 있지 않고, 매 경제 뉴스 기사마다 직접 라벨링 할 수 없으므로 뉴스 데이터를 직접 학습하는 모델을 만들기는 어려웠다.

3. 따라서 외부에서 pre-trained된 모델을 불러와서 뉴스 데이터를 라벨링 하는 방법을 생각했다.

4. Bert 모델을 기반으로 한 모델들 중 한국어로 학습시킨 Klue-Bert 모델에 대해 학습했다. </br>
[klue-bert모델](https://huggingface.co/klue)

5. 모델링을 담당하는 팀원과 담당을 나누어 팀원은 klue-bert-base모델, 나는 Klue-Roberta-Large 모델을 사용하기로 했다.

### Klue-Roberta 학습 내용

1. klue-roberta 모델의 특징
    - KLUE(Korean Language Understanding Evaluation) 데이터셋으로 사전 학습된 RoBERTa 기반 모델이다.
    - RoBERTa는 원래 BERT의 개선 버전으로, 더 많은 데이터와 최적화된 학습 방법을 사용한다.
    - KLUE-RoBERTa는 한국어 텍스트에 대한 문맥 이해, 감성 분석, 개체명 인식, 관계 추출 등의 다양한 한국어 NLP 작업에 활용된다.
    - 한국어의 특성을 고려하여 학습되었기 때문에, 한국어 처리에 있어서 더 높은 성능을 보인다.

2. Hugging Face에서 다음과 같은 방법으로 사용할 수 있다.
```python
from transformers import AutoTokenizer, AutoModel

# 토크나이저와 모델 로드하기
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")  # 또는 "klue/roberta-large"
model = AutoModel.from_pretrained("klue/roberta-base")  # 또는 "klue/roberta-large"

# 예시 텍스트
text = "안녕하세요, 한국어 자연어 처리 모델입니다."

# 토큰화 및 입력 준비
inputs = tokenizer(text, return_tensors="pt")

# 모델에 입력 전달하여 출력 얻기
outputs = model(**inputs)

# 출력의 마지막 히든 스테이트 가져오기
last_hidden_states = outputs.last_hidden_state
```

3. Klue-Roberta 모델은 추가 데이터셋으로 파인 튜닝 가능하다.</br>
[추가 데이터셋](https://github.com/ukairia777/finance_sentiment_corpus/blob/main/finance_data.csv)

```python 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import load_dataset

# 1. 데이터 준비
# 여기서는 예시로 Hugging Face datasets 라이브러리 사용
# 실제로는 자신의 데이터셋을 준비해야 함
dataset = load_dataset("your_dataset")  # 또는 커스텀 데이터셋 로드

# 2. 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=len(dataset["train"].features["label"].names))

# 3. 데이터 전처리 함수
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 4. 데이터셋 전처리 적용
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. 학습 파라미터 설정
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 6. Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# 7. 모델 학습
trainer.train()

# 8. 모델 저장
model.save_pretrained("./my_finetuned_klue_roberta")
tokenizer.save_pretrained("./my_finetuned_klue_roberta")
```

### 신용 평가 모델링을 위한 데이터셋 탐색

1. 캐글에서 초기에 찾은 데이터셋이 있었으나 138개 파일의 메타데이터로 분산돼 구성되어 있어서 사용하기가 어려웠다.

2. 새롭게 찾은 데이터셋</br>
[신용 평가 데이터셋](https://www.kaggle.com/datasets/ajay1735/hmeq-data)
---

# 250305_수

### 뉴스 데이터 크롤링 및 피그마 와이어프레임 회의

- 뉴스 데이터 크롤링 속도가 느려 팀원 모두가 크롤링에 참여하기로 결정헀고, 팀장 주도 하에 프로젝트 와이어프레임에 대한 회의를 진행했다.

### 신용 평가 데이터셋 결정

- 신용 평가 모델링을 위해 새롭게 찾은 데이터셋은 주택담보대출에 대한 상환 여부 데이터여서 신용 평가 모델링을 하기에는 적절하지 않았다.

- 따라서 원래 찾았던 신용 평가 빅데이터를 사용하기로 결정했다.

### 신용 평가 빅데이터 분석을 위한 개인 학습

1. 데이터셋을 탐구해보니 DB 설계를 끝낸 데이터들을 DB를 파일로 만들어 놓은 형태였다.

2. 너무 대규모의 데이터여서 Colab에서 처리하려니 메모리 초과가 발생했다.

3. 팀원과 상의 후에 팀원은 카드 리볼빙 데이터셋을 찾아 분석하기로 했고 나는 새롭게 Hadoop 환경에서 Spark를 이용해서 데이터 처리를 하고 분석까지 진행하기로 결정했다.

4. Hadoop 환경과 Spark 모두 처음 써보는 프레임워크이기 때문에 환경 설정부터 사용하는 방법까지 새롭게 학습하기 시작했다다.

5. 윈도우 환경으로 진행하는 것보다 리눅스 환경으로 진행하는 것이 더 의미 있다고 판단해서 Docker 컨테이너 위에서 작업하기로 했다.
