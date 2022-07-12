# pandas는 데이터 조작 및 분석을 위한 Python 프로그래밍
# CSV, Excel, JSON등의 데이터를 읽고 원하는 데이터 형식으로 변환
# 내부적으로 c언어로 돌아감, python은 다른 프로그래밍 언어와 잘 붙는다
# import pandas as pd
from pandas import Series, DataFrame

# Series 객체
# 일차원 배열같은 자료구조 객체
obj = Series([3,22,34,11])
obj

print(obj.values)
print(obj.index)
# 리스트가 아니다, 인덱스는 객체로 처리된다

# index가 보여서 지정 가능
obj2 = Series([4,5,6,2], index=['c','d','e','f'])
# 내가 인덱스를 붙여준것이다
obj2

# 인덱싱 해보기
obj2['c']

# 한 번에 불러오기
obj2[['c','d','f']]

# 각 요소별 연산
# 배열 연산은 각 요소끼리 한다
obj2 * 2
# 배열 연산에는 브로드케스팅 이라는 개념이 있다
# 배열 연산에 맞는 형식으로 만들어지기 때문에

# 딕셔너리와 거의 유사해서 대체가 가능하다
data = {'park':1000, 'kang':2000,'kim':3000}
# 딕셔너리로 만든 데이터를 판다스 객체로 만들어야 판다스 연산이 가능하다
# 딕셔너리와 판다스 객체가 매우 유사해서 아래와 같이 쉽게 만들 수 있다
obj3 = Series(data)
obj3

name = ['woo','hong','park']
obj4 = Series(data, index=name)
obj4

obj3

obj3.name = "최고득점"
obj3

# DataFrame 자료구조 객체
x = DataFrame([
  [1,2,3],
  [4,5,6],
  [7,8,9]
])
x

data = {
    'city' : ['서울','부산','광주','대구'],
    'year' : [2000,2001,2002,2002],
    'pop' : [4000, 2000, 1000, 1000]
}
# json 형식을 dataFrame으로 바꿀 수 있다

df = DataFrame(data)
df

df = DataFrame(data,columns=['year','city','pop','debt'], index=['one','two','three','four'])
df

# 인덱싱
df['city']
# 자동으로 Series로 빼준다

# 행 단위로 추출
df.loc['three']

# 값 삽입
df['debt'] = 1000
df

# 연속된 값 넣기
import numpy as np
df['dept'] = np.arange(4)
df 

# Series 를 이용해서 값 삽입
val = Series([1000,2000,3000,4000], index=['one','two','three','four'])
df['debt'] = val
df

# 값 삽입 = 연산의 결과로 t/f를 삽입  
df['cap'] = df.city == '서울'
df

# 전치 행렬
df.T

# 데이터만 추출
df.values