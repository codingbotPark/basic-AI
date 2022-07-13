# 대용량의 데이터가 필요
# 3v = 규모(value), 속도(velocity), 다양성(variety)
# 머신러닝은 알고리즘이 아닌, 데이터 학습을 통해 실행 동작이 바뀐다
# sklearn을 사용

# datasets.load_iris() 분류용도, 붓꽃에 대한 피러를 가진 데이터 세트
# 사이킷런을 이용해 붓꽃 품종을 분류하는 인공지능 모델


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터를 로드
iris = load_iris()
iris

# iris.data 는 피처로 된 데이터 (ndarray객체)
iris_data = iris.data

iris_data


# 지도학습을 선택했기 때문에 데이터와 정답을 줘야한다
iris_label = iris.target

iris_label_name = iris.target_names

print(iris_label)
print(iris_label_name) 
# 데이터 프레임으로 만들어서 관리를 쉽게 한다

# 데이터 프레임으로 바로 반환할 수 있는 데이터 또는 내가 반환해서 데이터프레임으로 사용할 수 있다

# 데이터 세트를 데이터프레임으로 변환
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names) # 매개변수로 공부시킬 데이터
iris_df['label'] = iris.target
iris_df.head()

# 학습용 데이터와 테스트용 데이터를 분리
train_test_split(iris_data, iris_label, test_size=0.2,random_state =11) 
# 테스트 데이터를 얼마만큼 떼어낼 것이냐(0.2 는 8대 2)
# random_state는 seed값을 고정해준다(학습용 데이터와 테스트용 데이터를 분리할 때 랜덤으로 진행되는데, seed값을 준다)

# 학습용 데이터와 테스트용 데이터를 분리
X_train, X_test, y_train,y_test = train_test_split(iris_data,iris_label,test_size=0.2,random_state=11)

# 인공지능 모델 알고리즘 선택
dt_clf = DecisionTreeClassifier(random_state = 11)

# 학습 수행
dt_clf.fit(X_train,y_train)

# 학습이 완료된 모델에 대해 평가를 수행
# 학습이 완료된 모델이 새로운 데이터를 가지고 분류를 수행
pred = dt_clf.predict(X_test)
pred

# 위 수행한 결과를 바탕으로 만든 모델의 성능을 평가
from sklearn.metrics import accuracy_score

print("예측 정확도 : {0:.4f}".format(accuracy_score(y_test,pred)))

# 프로세스 정리
# 데이터 로딩
# 데이터 탐색
# 데이터 세트 분리 (학습 데이터 : 테스트 데이터)
# 모델 학습 (학습데이터(feature)를 기반으로 ML알고리즘을 적용)
# 예측 수행 (학습된 ML모델을 이용)
# 평가 (예측 수행을 통해 예측된 결과값과 테스트데이터의 값 비교)

# 데이터 전처리
# Garbage In, Garbage Out
# 사이킷 런의 M알고리즘에 적용하기 전 데이터에 대해 미리 처리

# 결손값, NaN, NULL값은 허용되지 않는다
# Drop
# 대체값 선정

# 데이터 인코딩
# * 레이블 인코딩 = 카테고리 피처를 코드형 숫자값으로 바꾸는 방식

from sklearn.preprocessing import LabelEncoder

items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','컴퓨터','믹서','믹서']

encoder = LabelEncoder()
encoder.fit(items)

labels = encoder.transform(items)
print("인코딩 반환값 : ",labels)

# 이렇게 인코딩을 하면, 나만 그 값이 뭔지 알 수 있다,
# 그래서 원래 값이 뭔지까지 아렬줄 수 있다

print('인코딩 클래스 : ',encoder.classes_)

  # 원본 정보가 있으므로 디코딩 가능
  print('디코딩 값',encoder.inverse_transform([0 ,1 ,4 ,5 ,3 ,5 ,2 ,2]))

  # 레이블 인코딩은 레이블이 숫자로 증가하는 특성, 그러나 특정 ML알고리즘은 숫자 특성에 영향을 받는다(ex) 높은 숫자를 가중치)
# 그래서 이런 ML알고리즘에는 레이블 인코딩 사용을 지양
from sklearn.preprocessing import OneHotEncoder

items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']

# 먼저 숫자갓으로 변경, Label Encoder 사용
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
labels = labels.reshape(-1,1)
labels

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print('원 핫 인코딩 데이터')
print(oh_labels.toarray())
print('원 핫 인코딩 데이터 차원')
print(oh_labels.shape)

df = pd.DataFrame({'item':['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']})
pd.get_dummies(df)

# 피처 스케일링과 정규화
# 피처 스케일링 = 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
# 피처 스케일링의 대표적 작업 = 표준화/

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_data = iris.data 
# iris는 2차원 배열이기 때문에 
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df.head()  

# 평균값
iris_df.mean()

# 분산값
iris_df.var()

# StandardScalar = 표준화를 쉽게 지원하기 위한 사이킷런 클래스
from sklearn.preprocessing import StandardScaler

# 객체생성
scaler = StandardScaler()

scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_scaled_df = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)

print(iris_scaled_df.mean())
print('-------------')
print(iris_scaled_df.var()) 

# svm, 선형회귀, 로지스틱 회귀 알고리즘 등은 데이터가 가우시안 분포를 가지고 있다고 가정을 하고 설계되어 있는 알고리즘
# 사전에 표준화 작업을 하는 것은 예측 성능 향상에 중요한 요소가 된다

# MinMaxScaler
# 데이터를 0과 1사이의 범위 값으로 변환, 가우시안 분포가 아닐 경우 유용

from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()

# 데이터셋 변환
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# 결과를 데이터프레임 형태로 만들기
iris_scaled_df = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)


# 각 피처에 대햐여 평균값, 분산값 추출
print(iris_scaled_df.mean())
print('-----------')

# 평가, 이 모델이 얼마나 공부를 했는지
# 실제 데이터와 예측 데이터가 얼마나 같은지, 직관적으로 모델 예측 성능을 나타내는 지표, 데이터의 왜곡이 있을 수 있기 때문에 이 요소 하나만으로 판단하면 안 된다, 

# 오차행렬(혼돈행렬)
from IPython.display import Image

from google.colab import files
files.upload()

# 오차행렬은 학습된 분류 모델이 예측을 수행하며 얼마나 헷갈리고 있는지 보여준다
# TP = 예측값을 positive 1, 실제값이 positive 1
# TN = 예측값을 negative 0, 실제값이 negative 0
# FP = 예측값이 positive 1, 실제값이 negative 0
# FN = 예측값이 negative 0, 실제값이 positive 1
# 정확도 = (TP + TN) / (TP + TN + FP + FN)

# 정밀도와 재현율
# 업무 특성에 따라 특정 평가 지표가 더 중요한 경우가 있다 (재현율이 상대적으로 더 중요한 경우)
# 재현율이 더 중요한 경우 ex) 암 환자 판별, 금융 사기 적발  

# ROC Curve 와 AUC 
# ROC = 수신자 판단 곡선

# 피마 지역 인디언 당뇨병 예측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 데이터 로딩
diabetes_data = pd.read_csv('diabetes.csv')
diabetes_data.head()

# outcome을 label로 처리
diabetes_data.Outcome.value_counts()

# 데이터 탐색
diabetes_data.info()

# 평가 함수
def get_clf_eval(y_test,pred=None, pred_proba=None):
  confusion = confusion_matrix(y_test,pred)
  accuracy = accuracy_score(y_test,pred)
  precision = precision_score(y_test,pred)
  recall = recall_score(y_test,pred)
  f1 = f1_score(y_test,pred)
  roc_auc = roc_auc_score(y_test,pred_proba)

  # 오차행렬
  print('confusion matrix')
  print(confusion)
  print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현률 : {2:.4f}, f1 score : {3:.4f}, AUC : {4:.4f}'.format(accuracy,precision,recall,f1,roc_auc))

# 학습 밑 예측 시켜보기
features = diabetes_data.iloc[:,:-1]
label = diabetes_data.iloc[:, -1]

# stratify 는 원본 데이터의 벨런스대로 test_size를 뗀다
X_train,X_test,y_train,y_test = train_test_split(features, label, test_size = 0.2, random_state=156,stratify=label)

lr_clf =  LogisticRegression(max_iter=1000)
lr_clf.fit(X_train,y_train)

pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,-1]
get_clf_eval