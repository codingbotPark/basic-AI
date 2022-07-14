# 딥러닝 = 신경망을 층층히 쌓아서 문제를 해결하는 기법의 총칭
#
# 퍼셉트론 = 인공뉴런

# AND 게이트
import numpy as np
def AND(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.8
  tmp = w1* x1 + w2 * x2
  if tmp <= theta:
    return 0;
  elif tmp > theta:
    return 1

# 둘 다 1이고, 두 데이터 값이 임계값을 넘을 때 참

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))

# tensorFlow의 장점 = 편하다, 단점 = 안보인다

#손글씨 숫자 데이터 인식하는 딥러닝 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로딩 (머신러닝에서는 test데이터를 따로 줘야한다)
(X_train, y_train),(X_test,y_test) = mnist.load_data()
print("x_train shape",X_train.shape)
print("y_train shape",y_train.shape)
print("x_test shape",X_test.shape)
print("y_test shape",y_test.shape)
# 종속변수는 소문자

# 딥러닝 모델이 원하는 형태로 데이터를 전처리
# 정규화(값 차이가 너무 크기 때문에)
# 백터화
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

# 부동 소수점화
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 정규화
X_train /= 255
X_test /= 255
print('X training matrix shape',X_train.shape)
print('X testing matrix shape',X_test.shape)

# 레이블 데이터 원-핫 인코딩 처리
y_train=to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
print('Y training matrix shape',y_train.shape)
print('Y testing matrix shape',y_test.shape)

y_train[0] # 원래는 5가 적혀있었다

# 모델 생성 - 순전파
model = Sequential()
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

# 머신러닝 손실함수와 딥러닝 손실함수는 다르다
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 학습
model.fit(X_train,y_train,batch_size=128,epochs=10,verbose=1)

# 새로운 데이터를 통해 모델의 현재 수준을 평가
score = model.evaluate(X_test,y_test)
print('Test score:',score[0])
print('Test accuracy:',score[1])

import keras

# Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full), (X_test_full,y_test_full) = fashion_mnist.load_data()

# 데이터 살펴보기
X_train_full.shape

# 훈련데이터, 검증데이터로 분리 - 정규화
X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test_full = X_test_full / 255.0

import matplotlib as mpl
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)

import matplotlib.pyplot as plt
plt.imshow(X_train[0],cmap='binary')
plt.axis('off')
plt.show()

# 출력될 클래스 이름 지정
class_name = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

class_name[y_train[0]]

# 데이터 셋에 있는 이미지를 샘플로 확인
n_rows = 4
n_columns = 10

plt.figure(figsize=(n_columns * 1.2, n_rows * 1.2))

for row in range(n_rows):
  for col in range(n_columns):
    index = n_columns * row + col
    plt.subplot(n_rows,n_columns,index+1)
    plt.imshow(X_train[index], cmap="binary",interpolation="nearest")
    plt.axis('off')
    plt.title(class_name[y_train_full[index]], fontsize=12)

  plt.subplots_adjust(wspace=0.2,hspace=0.5)
  plt.show()