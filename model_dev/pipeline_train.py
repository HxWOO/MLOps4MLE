import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# 1. get data
X, y = load_iris(return_X_y=True, as_frame=True)
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
# 전이랑 다르게 pipeline 한번에 저장 가능
joblib.dump(model_pipeline, "model_pipeline.joblib")