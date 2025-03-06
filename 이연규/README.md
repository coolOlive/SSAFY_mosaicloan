### 목차
1. [250304_화](#250304_화)
2. [250305_수](#250305_수)
3. [250306_목](#250306_목)
---
<details>
  <summary><h1>250304_화</h1></summary>

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

</details>

<details>
  <summary><h1>250305_수</h1></summary>

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

</details>

<details>
  <summary><h1>250306_목</h1></summary>

### Spark 개인 학습

- Docker를 이용한 Spark 사용 환경 설정

    1. 설치
     - Docker Desktop, Spark, Jupyter Notebook, JDK21 

    2. 파일
    - docker-compose.yml 작성
    ```docker
    services:
    spark-master:
        image: bitnami/spark:3.3.0
        container_name: spark-master
        environment:
        - SPARK_MODE=master
        - SPARK_RPC_AUTHENTICATION_ENABLED=no
        - SPARK_RPC_ENCRYPTION_ENABLED=no
        - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
        - SPARK_SSL_ENABLED=no
        ports:
        - '8080:8080'  # 마스터 웹 UI
        - '7077:7077'  # 마스터 포트
        volumes:
        - ./data:/home/jovyan/data

    spark-worker-1:
        image: bitnami/spark:3.3.0
        container_name: spark-worker-1
        environment:
        - SPARK_MODE=worker
        - SPARK_MASTER_URL=spark://spark-master:7077
        - SPARK_WORKER_MEMORY=8G
        - SPARK_WORKER_CORES=3
        - SPARK_RPC_AUTHENTICATION_ENABLED=no
        - SPARK_RPC_ENCRYPTION_ENABLED=no
        - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
        - SPARK_SSL_ENABLED=no
        volumes:
        - ./data:/home/jovyan/data
        depends_on:
        - spark-master

    spark-worker-2:
        image: bitnami/spark:3.3.0
        container_name: spark-worker-2
        environment:
        - SPARK_MODE=worker
        - SPARK_MASTER_URL=spark://spark-master:7077
        - SPARK_WORKER_MEMORY=8G
        - SPARK_WORKER_CORES=3
        - SPARK_RPC_AUTHENTICATION_ENABLED=no
        - SPARK_RPC_ENCRYPTION_ENABLED=no
        - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
        - SPARK_SSL_ENABLED=no
        volumes:
        - ./data:/home/jovyan/data
        depends_on:
        - spark-master

    spark-worker-3:
        image: bitnami/spark:3.3.0
        container_name: spark-worker-3
        environment:
        - SPARK_MODE=worker
        - SPARK_MASTER_URL=spark://spark-master:7077
        - SPARK_WORKER_MEMORY=8G
        - SPARK_WORKER_CORES=3
        - SPARK_RPC_AUTHENTICATION_ENABLED=no
        - SPARK_RPC_ENCRYPTION_ENABLED=no
        - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
        - SPARK_SSL_ENABLED=no
        volumes:
        - ./data:/home/jovyan/data
        depends_on:
        - spark-master

    jupyter:
        image: jupyter/pyspark-notebook:spark-3.3.0
        container_name: jupyter-pyspark
        environment:
        - JUPYTER_ENABLE_LAB=yes
        - SPARK_MASTER=spark://spark-master:7077 
        ports:
        - '8888:8888'  # Jupyter 노트북
        - '4040:4040'  # Spark 애플리케이션 UI
        volumes:
        - ./work:/home/jovyan/work
        - ./data:/home/jovyan/data
        depends_on:
        - spark-master

    ``` 
    3. docker-compose up 명령어 실행 후 로컬 url 접속

    4. spark session 생성
    ```python
        from pyspark.sql import SparkSession

        # 클러스터에 연결
        spark = SparkSession.builder \
            .appName("Distributed Analysis") \
            .master("spark://spark-master:7077") \
            .config("spark.executor.memory", "6g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.memory", "6g") \
            .config("spark.sql.shuffle.partitions", "30") \
            .getOrCreate()
    ```

    5. 데이터 분석 진행

#### Trouble Shooting

```
Py4JJavaError                             Traceback (most recent call last)
Cell In[2], line 2
      1 # 디렉토리 내의 모든 Parquet 파일을 한 번에 읽기
----> 2 df = spark.read.parquet("/home/jovyan/data/parquet_files/train/")
      4 # 스키마 확인
      5 df.printSchema()

File /usr/local/spark/python/pyspark/sql/session.py:1706, in SparkSession.read(self)
   1669 @property
   1670 def read(self) -> DataFrameReader:
   1671     """
   1672     Returns a :class:`DataFrameReader` that can be used to read data
   1673     in as a :class:`DataFrame`.
   (...)
   1704     +---+------------+
   1705     """
-> 1706     return DataFrameReader(self)

File /usr/local/spark/python/pyspark/sql/readwriter.py:70, in DataFrameReader.__init__(self, spark)
     69 def __init__(self, spark: "SparkSession"):
---> 70     self._jreader = spark._jsparkSession.read()
     71     self._spark = spark

File /usr/local/spark/python/lib/py4j-0.10.9.7-src.zip/py4j/java_gateway.py:1322, in JavaMember.__call__(self, *args)
   1316 command = proto.CALL_COMMAND_NAME +\
   1317     self.command_header +\
   1318     args_command +\
   1319     proto.END_COMMAND_PART
   1321 answer = self.gateway_client.send_command(command)
-> 1322 return_value = get_return_value(
   1323     answer, self.gateway_client, self.target_id, self.name)
   1325 for temp_arg in temp_args:
   1326     if hasattr(temp_arg, "_detach"):

File /usr/local/spark/python/pyspark/errors/exceptions/captured.py:179, in capture_sql_exception.<locals>.deco(*a, **kw)
    177 def deco(*a: Any, **kw: Any) -> Any:
    178     try:
--> 179         return f(*a, **kw)
    180     except Py4JJavaError as e:
    181         converted = convert_exception(e.java_exception)

File /usr/local/spark/python/lib/py4j-0.10.9.7-src.zip/py4j/protocol.py:326, in get_return_value(answer, gateway_client, target_id, name)
    324 value = OUTPUT_CONVERTER[type](answer[2:], gateway_client)
    325 if answer[1] == REFERENCE_TYPE:
--> 326     raise Py4JJavaError(
    327         "An error occurred while calling {0}{1}{2}.\n".
    328         format(target_id, ".", name), value)
    329 else:
    330     raise Py4JError(
    331         "An error occurred while calling {0}{1}{2}. Trace:\n{3}\n".
    332         format(target_id, ".", name, value))

Py4JJavaError: An error occurred while calling o33.read.
: java.lang.IllegalStateException: LiveListenerBus is stopped.
	at org.apache.spark.scheduler.LiveListenerBus.addToQueue(LiveListenerBus.scala:92)
	at org.apache.spark.scheduler.LiveListenerBus.addToStatusQueue(LiveListenerBus.scala:75)
	at org.apache.spark.sql.internal.SharedState.<init>(SharedState.scala:115)
	at org.apache.spark.sql.SparkSession.$anonfun$sharedState$1(SparkSession.scala:143)
	at scala.Option.getOrElse(Option.scala:189)
	at org.apache.spark.sql.SparkSession.sharedState$lzycompute(SparkSession.scala:143)
	at org.apache.spark.sql.SparkSession.sharedState(SparkSession.scala:142)
	at org.apache.spark.sql.SparkSession.$anonfun$sessionState$2(SparkSession.scala:162)
	at scala.Option.getOrElse(Option.scala:189)
	at org.apache.spark.sql.SparkSession.sessionState$lzycompute(SparkSession.scala:160)
	at org.apache.spark.sql.SparkSession.sessionState(SparkSession.scala:157)
	at org.apache.spark.sql.DataFrameReader.<init>(DataFrameReader.scala:699)
	at org.apache.spark.sql.SparkSession.read(SparkSession.scala:783)
	at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:77)
	at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.base/java.lang.reflect.Method.invoke(Method.java:568)
	at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
	at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:374)
	at py4j.Gateway.invoke(Gateway.java:282)
	at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
	at py4j.commands.CallCommand.execute(CallCommand.java:79)
	at py4j.ClientServerConnection.waitForCommands(ClientServerConnection.java:182)
	at py4j.ClientServerConnection.run(ClientServerConnection.java:106)
	at java.base/java.lang.Thread.run(Thread.java:833)
```
- LiveListenerBus is stopped. 에러는 spark session이 예기치 못하게 종료되었을 때 발생한다고 한다. 하지만 주피터 노트북 커널을 재실행하고 Docker를 재실행해서 spark session을 재생성해도 같은 오류가 발생했다.

- 여러가지를 확인해봤을 때 오류는 버전 불일치에서 일어난다는 것을 알게 되었다. Jupyter를 띄우는 Docker 노드에서는 Spark를 최신버전을 이용하고 master 노드와 worker 노드에서는 3.3.0 버전을 사용하고 있어서 발생한 문제였다.

- Jupyter의 Spark 버전을 3.3.0으로 바꿨지만 다른 오류가 발생하였다.

- 이 오류는 parquet 파일을 경로에서 찾지 못하는 오류였는데 원인은 Docker-compose 파일에서 Volumes에 등록된 data폴더 경로가 일치하지 않아서 생긴 문제였다.

- ./data : /home/jovyan/data 으로 모든 노드를 일치시켜준 뒤에 정상적으로 파일을 불러올 수 있었다.


### 브랜치 전략, 커밋/코딩 컨벤션 회의

</details>