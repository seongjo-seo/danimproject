## SURPRISE
import surprise
# import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.model_selection import train_test_split
import requests
import json
import os
import pickle
from sqlalchemy import create_engine
import pymysql


# engine = create_engine("mysql+mysqldb://root:changs213@localhost/danim")
# conn = engine.connect()
# con = pymysql.connect(
#     host="localhost", port=3306, user="root", password="changs213", db="danim"
# )
# cur = con.cursor()
#
# sql = "select * from travel_records"
#
# cur.execute(sql)
# rows = list(cur.fetchall())
# con.commit()
#
# raw = pd.DataFrame.from_records(rows, columns =['n_items', 'n_users','travel_date','rating'])
# raw = raw[['n_items', 'n_users','rating']]
#
# sql2 = "select * from travel_places"
#
# cur.execute(sql2)
# rows2 = list(cur.fetchall())
# con.commit()
#
# travels = pd.DataFrame(rows2, columns=['place_id','place_name','place_region1','place_region2','place_region3','issued_place','location_keyword','feeling','in_out'])
# travels = travels[['place_id', 'place_name', 'issued_place', 'place_region2', 'feeling', 'in_out']]
#
# sql3 = "select * from korea_weather"
#
# cur.execute(sql3)
# rows3 = list(cur.fetchall())
# con.commit()
#
# cities_weathers = pd.DataFrame(rows3, columns=['id','city','weather'])
# weathers = travels[['place_id', 'place_name', 'issued_place', 'place_region2', 'feeling', 'in_out']]

raw = pd.read_csv('./travel_records.csv')
# raw.drop_duplicates(inplace=True)

# swapping columns
raw = raw[['userid', 'place_id', 'point']]
raw.columns = ['n_users', 'n_items', 'rating']

# 모델 받아오기
with open('models/collabKNN.pkl', 'rb') as f:
    collabKNN = pickle.load(f)
with open('models/funkSVD.pkl', 'rb') as f:
    funkSVD = pickle.load(f)
# with open('models/coClus.pkl', 'rb') as f:
#     coClus = pickle.load(f)
# with open('models/slopeOne.pkl', 'rb') as f:
#     slopeOne = pickle.load(f)

travels = pd.read_csv('./travel_places.csv', encoding='cp949')
travels = travels[['place_id', 'place_name', 'issued_place', 'place_region2', 'feeling', 'in_out']]

cities_weathers = pd.read_csv('./korea_weather.csv')
# cities_weathers = pd.DataFrame(rows3, columns=['id','city','weather'])

def get_unseen_surprise(ratings, travels, userId):
    been_travels = ratings[ratings['n_users'] == userId]['n_items'].tolist()
    #     print(been_travels)
    total_travels = travels['place_id'].tolist()

    notbeen_travels = [travel for travel in total_travels if travel not in been_travels]
    print('가본 여행지 수:', len(been_travels), '추천대상 여행지:', len(notbeen_travels), \
          '전체 여행지:', len(total_travels))

    return notbeen_travels

def recomm_travel_by_surprise(algo, userid, notbeen_travels, top_n=10):
    # 알고리즘 객체의 predict() 메서드를 평점이 없는 여행지에 반복 수행한 후 결과를 list 객체로 저장
    predictions = [algo.predict(userid, int(place_id)) for place_id in notbeen_travels]
    raw_predicts = predictions.copy()

    # predictions list 객체는 surprise의 Predictions 객체를 원소로 가지고 있음.
    # 이를 est 값으로 정렬하기 위해서 아래의 sortkey_est 함수를 정의함.
    # sortkey_est 함수는 list 객체의 sort() 함수의 키 값으로 사용되어 정렬 수행.
    def sortkey_est(pred):
        return pred.est

    # sortkey_est( ) 반환값의 내림 차순으로 정렬 수행하고 top_n개의 최상위 값 추출.
    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]

    top_travel_ids = [int(pred.iid) for pred in top_predictions]
    top_travel_rating = [pred.est for pred in top_predictions]
    top_travel_rating = list(map(float, top_travel_rating))
    #     top_travel_titles = travels[travels.place_id.isin(top_travel_ids)]['place_name']
    top_travel_titles = []
    for ids in top_travel_ids:
        iid = travels[travels['place_id'] == ids].place_name.index[0]
        name = travels.loc[iid].place_name
        top_travel_titles.append(name)
    top_travel_preds = [(id, title, rating) for id, title, rating in
                        zip(top_travel_ids, top_travel_titles, top_travel_rating)]

    return top_travel_preds, raw_predicts

# 날씨, 기분
feeling = '나쁨'

def weather(city):
    apiKey = ""
    api = 'https://api.openweathermap.org/data/2.5/weather?q=' + city + '&appid=' + apiKey
    result = requests.get(api)
    result = json.loads(result.text)
    w_now = result['weather'][0]['main']

    return w_now

def final_result(user, weights, top_n):
    # 사용자가 가보지 않은 여행지 불러오기
    notbeen_travels = get_unseen_surprise(raw, travels, user)
    # 각각의 모델들로부터 예상 별점 가져오기
    top_travel_preds1, raw_predicts1 = recomm_travel_by_surprise(collabKNN, user, notbeen_travels, top_n)
    top_travel_preds2, raw_predicts2 = recomm_travel_by_surprise(funkSVD, user, notbeen_travels, top_n)
    # top_travel_preds3, raw_predicts3 = recomm_travel_by_surprise(coClus, user, notbeen_travels, top_n)
    # top_travel_preds4, raw_predicts4 = recomm_travel_by_surprise(slopeOne, user, notbeen_travels, top_n)

    raw_predicts_list = [raw_predicts1, raw_predicts2]#, raw_predicts3, raw_predicts4]

    rating_dict1 = {}
    rating_dict2 = {}
    # rating_dict3 = {}
    # rating_dict4 = {}

    rating_dicts = [rating_dict1, rating_dict2]#, rating_dict3, rating_dict4]

    # 딕셔너리에 여행지 아이디와 예상 별점 넣기
    for i in raw_predicts1:
        rating_dict1[i.iid] = i.est
    for i in raw_predicts2:
        rating_dict2[i.iid] = i.est
    # for i in raw_predicts3:
    #     rating_dict3[i.iid] = i.est
    # for i in raw_predicts4:
    #     rating_dict4[i.iid] = i.est

    # 재난지역
    disaster_list = list(travels[travels['issued_place'] == 1].place_id)

    # 재난지역에는 0.1점씩 가점
    for rating in rating_dicts:
        for key, value in rating.items():
            if key in disaster_list:
                rating[key] = value + 0.1

    # 가중치대로 별점 나누기
    result1 = {key: value * weights[0] for key, value in rating_dict1.items()}
    result2 = {key: value * weights[1] for key, value in rating_dict2.items()}
    # result3 = {key: value * weights[2] for key, value in rating_dict3.items()}
    # result4 = {key: value * weights[3] for key, value in rating_dict4.items()}

    results = [result1, result2]#, result3, result4]

    # 모델별 별점 합치기
    final_rating = {}
    for i in result1.keys():
        # 기분에 맞는 여행지 거르기
        if feeling in travels[travels.place_id == i].values[0][4]:
            # 비오면 야외 여행지 거르기
            test = travels[travels.place_id == 54].values[0][3]
            if (cities_weathers.index[(cities_weathers['city'] == test)].values[0] == 'Rain') & (
                    travels[travels.place_id == i].values[0][5] == 'out'):
                continue
            else:
                test = result1[i] + result2[i]# + result3[i] + result4[i]
                final_rating[i] = test
        else:
            continue

    # 상위 n개의 여행지 추리기
    d2 = sorted(final_rating.items(), key=lambda x: x[1], reverse=True)
    sorted_d2 = d2[:top_n]

    # top_n으로 추출된 여행지의 정보 추출. 여행지 아이디, 추천 예상 평점, 이름 추출
    top_travel_ids = [int(pred[0]) for pred in sorted_d2]
    # print(top_travel_ids)
    top_travel_rating = [float(pred[1]) for pred in sorted_d2]
    # print(top_travel_rating)
    top_travel_titles = []
    for ids in top_travel_ids:
        iid = travels[travels['place_id'] == ids].place_name.index[0]
        name = travels.loc[iid].place_name
        top_travel_titles.append(name)
    top_travel_preds = [(id, title, rating) for id, title, rating in
                        zip(top_travel_ids, top_travel_titles, top_travel_rating)]
    return top_travel_preds, raw_predicts1


weights = [0.3, 0.7]
user = 10961#1135#10905  # 1135
top_n, raw_predicts1 = final_result(user, weights, top_n=20)

for i in range(len(top_n)):
    print(top_n[i][1])

# con.close()
# conn.close()