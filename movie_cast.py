from warnings import simplefilter
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

pd.set_option('display.max_columns',None)

movie=pd.read_csv(r'C:\coding\movie_recommendation\myvenv\tmdb_5000_movies.csv')
credits=pd.read_csv(r'C:\coding\movie_recommendation\myvenv\tmdb_5000_credits.csv')
credits.columns=['id','title','cast','crew']
movie=pd.merge(credits,movie,on="id",suffixes=("_x",""))
print(movie.columns)


movies_df=movie[['id','title','genres','keywords','overview','cast','crew','vote_average','vote_count','popularity']]
#컬럼을 literal_eval에 적용
features=['cast']
for feature in features:
    movies_df[feature]=movies_df[feature].apply(literal_eval)

movies_df['cast']=movies_df['cast'].apply(lambda x: [y['name'] for y in x])
movies_df['cast_size']=movies_df['cast'].apply(lambda x: len(x))
print('---TMDB영화의 배우, 배우 수---')
print(movies_df['cast'],movies_df['cast_size'])
movies_df['cast']=movies_df['cast'].apply(lambda x: x[:3] if len(x)>3 else x)
print('---가장 영향력 있는 배우---')
print(movies_df['cast'])

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
#clean_data함수를 cast 필드에 적용
features = ['cast']

for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)
print('---전처리---')
print(movies_df['cast'])

def create_literal(x):
    return ' '.join(x['cast']) 
#gneres 칼럼의 개별요소를 공백문자로 구분하는 문자열로 변환해 genres_literal 칼럼에 저장
movies_df['cast_literal'] = movies_df.apply(create_literal, axis=1)

def get_movie(cast_name):
    is_cast=movies_df['cast_literal'].str.contains(cast_name)
    subset=movies_df[is_cast]
    return subset[['title','vote_average','cast']].sort_values('vote_average',ascending=False)
print(get_movie("zoesaldana"))

