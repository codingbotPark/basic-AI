from random import random
import numpy as np

a = np.array([1,2,3])
b = [1,2,3]

print(a)
print(b)

print(type(a))
print(type(b))

a.shape

temp = 3
print(temp)

# 차원
# 1차원
# 2차원 ( rank가 2인 배열 생성 )
b = np.array([[1,2,3],[4,5,6]])

# arange
arr = np.arange(12)
arr

arr = np.arange(12).reshape(3,4); arr

# 특수한 행렬
np.zeros((3,4))

# 정방행렬
np.ones((3,3))

np.full((3,3),10)

# 기술 통계
x = np.array([1,2,56,24,-24,32424,234,2,7,8,8,35,435,345,73,5])
# 평균
np.mean(x)
# 분산
np.var(x)
# 최대값, 최소값, 중앙값
print(np.max(x),np.min(x),np.median(x))
# 배열 인덱싱
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]); a

# [1,2
# 5,6] 추출
b = a[:2,:2]; b
# [2,3
# 6,7]
c = a[:2,1:3]; c
b[0,0] = 100; b
a # 얕은 복사
d = a[1:2, :]; d
e = a[:,1:2]; e

d = a[1:2, :]; d
e = a[:,1:2]; e

# 난수 (Random Number)
# numpy 안의 random 함수에 있다
np.random.rand(5)

# 같은 seed, 씨앗값을 주면 같은 결과가 나오게 된다
np.random.seed(0)
np.random.rand(5)
# 즉 완벽한 난수를 만들려면 seed를 계속 변경시켜줘야한다

np.random.seed(2)
np.random.rand(5)

# np.random.randn() = 가우시안 표준
np.random.randn(5)

# 선생님 코드
np.random.randint(1,46,[5,6])

# 데이터 샘플링 : 이미 있는 데이터 집합에서 무작위로 선택하는 것
# np.random.choice(원본데이터, size, replace = True/False, p)
np.random.choice(5,5,replace=False)

np.random.choice(5,3,replace=False)

# 나올 수 있는 값이 현재 5개 있기 때문에 p에 5개의 요소별로 확률을 제어할 수 있다
np.random.choice(5,10,replace=True,p=[0.1,0,0,0.3,0.6])

# 난수 생성
x = np.arange(10)
np.random.shuffle(x)
print(x)