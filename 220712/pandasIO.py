# 데이터 입출력
import pandas as pd
from pandas import Series,DataFrame

from google.colab import files
files.upload()

# pandas IO는 제대로 듣지 않았다...


# 여러 개의 DataFrame을 엑셀 파일 하나에 각각(sheet를 지정) 저장
data1 = {
    'name' : ['Jerry','Riah','Paul'],
    'algo' : ['A', 'A+','B'],
    'basic' : ['B','B+','B'],
    'python' : ['B+','C','C+'],
}

data2 = {
    'c0' : [1,2,3],
    'c1' : [4,5,6],
    'c2' : [7,8,9],
    'c3' : [10,11,12],
    'c4' : [13,14,15],
}

df1 = DataFrame(data1)
df1.set_index('name',inplace=True)

df2 = DataFrame(data2)
df2.set_index('c0',inplace=True)

# 두 개의 데이터프레임 객체 생성, 헤더를 설정했다
# ExcelWriter 와 함께 사용
writer = pd.ExcelWriter('./df_excelWriter.xlsx')
df1.to_excel(writer,sheet_name="sheet1")
df2.to_excel(writer,sheet_name="sheet2")
writer.save()

# 파일을 다운로드
files.download('./df_excelWriter.xlsx')


# 데이터 사전처리, 전처리
# NaN, 누락데이터 처리
import seaborn as sns

df = sns.load_dataset('titanic')
df.head()

df.info()

# deck 열의 NaN 개수 확인
df.deck.value_counts() # 기본적으로 NaN 을 제외하고 센다
df.deck.value_counts(dropna=False)

df.deck.isnull().sum()

df.deck.isnull()

# 누락 데이터 제거 thresh로 기준점을 설정할 수 있다
df_tresh = df.dropna(thresh=500,axis=1)
df_tresh.columns