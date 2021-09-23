from warnings import simplefilter
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movie=pd.read_csv(r'C:\coding\movie_recommendation\myvenv\tmdb_5000_movies.csv')
movies_df=movie[['id','title','genres','keywords','vote_average','vote_count','popularity','overview']]
#장르 컬럼을 literal_eval에 적용
features=['genres']
for feature in features:
    movies_df[feature]=movies_df[feature].apply(literal_eval)

movies_df['genres']=movies_df['genres'].apply(lambda x: [y['name'] for y in x])
print('---TMDB영화의 장르---')
print(movies_df['genres'])

#모든 문자를 소문자로 변환, 공백 제거
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
         #문자열이 존재시 아래의 작업 수행, 없으면 빈 문자열 반환
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
#clean_data함수를 genres 필드에 적용
features = ['genres']

for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)
   

def create_literal(x):
    return ' '.join(x['genres']) 
#gneres 칼럼의 개별요소를 공백문자로 구분하는 문자열로 변환해 genres_literal 칼럼에 저장
movies_df['genres_literal'] = movies_df.apply(create_literal, axis=1)
print('---genres_literal : 데이터 전처리 및 문자열로 변환한 결과---')
print(movies_df['genres_literal'].head(50))

#유사도행렬
cnt_vect=CountVectorizer(min_df=0,ngram_range=(1,2))
genres_mat=cnt_vect.fit_transform(movies_df['genres_literal']) 
print('---피처백터행렬---')
print(genres_mat.shape)
#genre_sim객체는 행별 유사도 정보를 가짐
genre_sim=cosine_similarity(genres_mat,genres_mat)
print('---코사인 유사도 행렬---')
print(genre_sim[:100])

#argesort이용해 유사도가 높은 순으로 정리된 genre_sim객체의 비교 행 위치 인덱스값추출
genre_sim_idx=genre_sim.argsort()[:, ::-1]
print("---유사도 높은 영화의 인덱스---")
print(genre_sim_idx[:1]) 
#특정영화를 기준으로 선정해 유사도가 높은 영화 반환 하는 함수
def find_sim_movie(df,sorted_idx,title_name,top_n=10):
    title_movie=df[df['title']==title_name]
    title_movie_idx=title_movie.index.values
    similar_indexes=sorted_idx[title_movie_idx, :top_n]
    print("---유사도가 가장 높은 상위 10개의 영화의 인덱스---")
    print(similar_indexes)

    top_sim_idx=similar_indexes.reshape(-1)
    return df.iloc[top_sim_idx].sort_values('vote_average',ascending=False)
similar_movie=find_sim_movie(movies_df,genre_sim_idx,"Titanic",10)
print(similar_movie[['title','vote_average']])
print(similar_movie['genres'])