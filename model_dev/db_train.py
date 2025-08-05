import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# 1. get data
# id 기준으로 100개 가져옴
db_connect = psycopg2.connect(host="localhost", database="mydatabase", user="hyun", password="1234")
df = pd.read_sql("SELECT * FROM iris_data ORDER BY id DESC LIMIT 100", db_connect)
X = df.drop(["id", "timestamp", "target"], axis="columns")
y = df["target"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2024)

# 2. model development and train
# 모델 훈련 파이프 라인 작성 / (모델 이름, 모델 객체)의 튜플 리스트
model_pipeline = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])

# 파이프라인 학습, 일반적인 scikit-learn 의 모델처럼 진행
model_pipeline.fit(X_train, y_train)

# 학습이 완료된 파이프라인은 바로 예측을 하거나 각 단계별로 진행해 볼 수 있다
# 예를 들어서 scaler 만 사용하고 싶은 경우에는 아래처럼 할 수 있다
# print(model_pipeline[0].transform(X_train[:1]))

train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# 3. save model
joblib.dump(model_pipeline, "db_pipeline.joblib")

# 4. save data / 현재 data_generator를 통해 DB에 계속 data가 쌓이고 있기 때문에 데이터를 저장해서 validate할때 활용
df.to_csv("data.csv", index=False)
