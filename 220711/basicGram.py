# 숫자형
num1 = 5
num2 = 5.5
print(num1 + num2)

# 연산
num3 = 5 * 1
num4 = 5 * 1.0
print(num3,num4)

print('Jack\'s favo')

"""아우렐리우스
만물은 변화다
우리의 삶이란 우리의 생각이 변화를 만드는 (과정)이다"""

# 리스트형 <->  배열과 다르다
a = []
b = [1,2,3]

# 딕셔너리형
game = {
    "가위":"보",
    "보" :"가위"
}

print(game['가위'])

# 튜플
# 튜플은 한 번 지정한 값의 변경을 최소화 하고 싶을 때 사용
tuple1 = (1,2,3,4,5)
print(tuple1)

# 형변환
list1 = [1,2,3,4,5]
tuple3 = tuple(list1)
print(tuple3)

# 튜플을 이용한(콤마만 있으면 된다)
x = 5
y = 10
x,y = y,x
print(x,y)

# 문자열formating

list1 = [1,2,3,4,5]
for i in list1:
  print(i)

  # enumerate는 index번호를 추출해주는 함수이다
list1 = [1,2,3,4,5]
for i,j in enumerate(list1):
  print("{} 번째 값은 {} 이다".format(i,j))

for i in enumerate(list1):
  print("{} 번째 값은 {} 이다".format(*i))

# 딕셔너리
# 딕셔너리의 함수에는
# items = key와 value모두 리턴
dict = {'python':100, 'java':90,'jsp':90}

for key,val in dict.items():
  print("{} 점수는 {} 이다".format(key,val))
for i in dict.items():
  print("{} 점수는 {} 이다".format(i[0],i[1]))
for i in dict.items():
  print("{} 점수는 {} 이다".format(*i))

def tuple_good():
  for i in range(100):
    return i
list1 = tuple_good()
print(list1)
# 결과값 하나만 리턴

def tuple_good2():
  return 1,2
num1, num2 = tuple_good2()
print(num1, num2)

#numpy 는 백터 산술 연산, 다차원 배열, 표준 수학 함수, 선형대수, 난수를 다룬다
# pip로 파이썬 라이브러리를 사용할 수 있다
# 구글 클라우드에는 기본적으로 포함되어있다