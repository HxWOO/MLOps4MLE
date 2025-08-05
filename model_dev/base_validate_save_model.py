from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. reproduce data
# 학습 데이터 재현
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=2025)

# 2. load model
# 저장된 모델 불러오기
scaler_load: StandardScaler = joblib.load("scaler.joblib")
classifier_load: SVC = joblib.load("classifier.joblib")

# 3. validate
# 데이터 스케일링
scaled_X_train = scaler_load.transform(X_train)
scaled_X_valid = scaler_load.transform(X_valid)

# 예측 진행
load_train_pred = classifier_load.predict(scaled_X_train)
load_valid_pred = classifier_load.predict(scaled_X_valid)

# 정확도 계산, 이전 base_train 에서와 같은지 확인
load_train_acc = accuracy_score(y_true=y_train, y_pred=load_train_pred)
load_valid_acc = accuracy_score(y_true=y_valid, y_pred=load_valid_pred)

print("Load Model Train Accuracy :", load_train_acc)
print("Load Model Valid Accuracy :", load_valid_acc)