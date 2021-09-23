from warnings import simplefilter
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns',None)

movie=pd.read_csv(r'C:\coding\movie_recommendation\myvenv\tmdb_5000_movies.csv')
credits=pd.read_csv(r'C:\coding\movie_recommendation\myvenv\tmdb_5000_credits.csv')
credits.columns=['id','title','cast','crew']
movie=pd.merge(credits,movie,on="id",suffixes=("_x",""))
print(movie.columns)


movies_df=movie[['id','title','genres','keywords','overview','cast','crew','vote_average','vote_count','popularity']]
#컬럼을 literal_eval에 적용
features=['crew']
for feature in features:
    movies_df[feature]=movies_df[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job']=="Director":
            return i['name']
    return np.nan
movies_df['director']=movies_df['crew'].apply(get_director)

print('---TMDB영화의 감독---')
print(movies_df['director'])

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
#clean_data함수를 director 필드에 적용
features = ['director']

for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)
print('---전처리---')
print(movies_df['director'])

def get_movie(director_name):
    is_director=movies_df['director'].str.contains(director_name)
    subset=movies_df[is_director]
    return subset[['title','vote_average','director']].sort_values('vote_average',ascending=False)
print(get_movie("jamescameron"))


# #유사도행렬
# cnt_vect=CountVectorizer(min_df=0,ngram_range=(1,1))
# director_mat=cnt_vect.fit_transform(movies_df['director']) 
# print('---피처백터행렬---')
# print(director_mat.shape)
# #director_mat객체는 행별 유사도 정보를 가짐
# director_sim=cosine_similarity(director_mat,director_mat)
# print('---코사인 유사도 행렬---')
# print(director_sim[:100])

# #argesort이용해 유사도가 높은 순으로 정리된 genre_sim객체의 비교 행 위치 인덱스값추출
# director_sim_idx=director_sim.argsort()[:, ::-1]
# print("---유사도 높은 영화의 인덱스---")
# print(director_sim_idx[:1]) 
# #특정영화를 기준으로 선정해 유사도가 높은 영화 반환 하는 함수
# def find_sim_movie(df,sorted_idx,title_name,top_n=10):
#     title_movie=df[df['title']==title_name]
#     title_movie_idx=title_movie.index.values
#     similar_indexes=sorted_idx[title_movie_idx, :top_n]
#     print("---유사도가 가장 높은 상위 10개의 영화의 인덱스---")
#     print(similar_indexes)

#     top_sim_idx=similar_indexes.reshape(-1)
#     return df.iloc[top_sim_idx].sort_values('vote_average',ascending=False)
# similar_movie=find_sim_movie(movies_df,director_sim_idx,"Avatar",10)
# print(similar_movie[['title','vote_average']])
# print(similar_movie['director'])






