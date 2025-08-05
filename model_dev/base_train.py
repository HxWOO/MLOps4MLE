from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. get data
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2025)

# 2. model development and train
# fit 을 통해 scaler를 학습한 후 transform을 이용해 데이터를 scaling
scaler = StandardScaler()
classifier = SVC()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_valid = scaler.transform(X_valid)

# fit 함수를 이용해 모델을 학습
classifier.fit(scaled_X_train, y_train)

# SVC 를 scaling 이 된 데이터를 사용했기 때문에 scaled_X_train 과 scaled_X_valid 를 통해 예측
# 학습 데이터(train_pred)와 평가 데이터(valid_pred)에 대해서 예측을 진행
train_pred = classifier.predict(scaled_X_train)
valid_pred = classifier.predict(scaled_X_valid)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
valid_acc = accuracy_score(y_true=y_valid, y_pred=valid_pred)
# 학습된 모델이 정상적으로 예측하기 위해서는 데이터를 변환할 때 사용한 scaler 도 같이 저장되어야 함

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# 3. save model
# joblib 패키지를 이용해 모델 저장 (scaler, classifier)
joblib.dump(scaler, "scaler.joblib")
joblib.dump(classifier, "classifier.joblib")