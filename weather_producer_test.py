from kafka import KafkaProducer
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import time


#爬氣象資料
def get_weather_data(url):
    response = requests.get(url)
    
    # 抓資料
    if response.status_code == 200:
        data = response.json()
        
        # 提取 Station 列表中的資訊
        stations = data['records']['Station']
        
        # 創建一個空的 DataFrame
        df = stations
        df = pd.json_normalize(df)

        df = df.drop(['StationId','GeoInfo.Coordinates', 'GeoInfo.StationAltitude','GeoInfo.CountyName', 'GeoInfo.TownName', 'GeoInfo.TownCode','WeatherElement.GustInfo.Occurred_at.DateTime'], axis = 1)
        
        columns_names = ['StationName','ObsTime','CountyCode','Weather','Now','WindDirection','WindSpeed','AirTemperature','RelativeHumidity','AirPressure','PeakGustSpeed','WindDirection_1','DailyHighTemperature','DailyHighTemperatureOccurred_at','DailyLowAirTemperature','DailyLowTemperatureOccurred_at']
        df.columns = columns_names
        
        df = df.drop(['WindDirection_1', 'Now'], axis = 1)
        
        #取得obs_time並處理字串
        obs_time = df.at[0,'ObsTime'][:19].replace('T', ' ')
        
        df['ObsTime'] = obs_time
        
        return df, obs_time
    else:
        print(f"Error: {response.status_code}")
        return None

#建立producer
def create_kafka_producer(bootstrap_servers='localhost:9092'):
    return KafkaProducer(bootstrap_servers='localhost:9092', 
              value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'))


def main():
    rain_url = "https://opendata.cwb.gov.tw/api/v1/rest/datastore/O-A0002-001?Authorization=CWB-362C1548-B175-4BE4-AEDD-3978655ED307"
    weather_url = "https://opendata.cwb.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization=CWB-362C1548-B175-4BE4-AEDD-3978655ED307"
    #建立producer
    producer = create_kafka_producer()
    
    '''
    #爬雨量、氣象資料
    weather_df, weather_obs_time = get_weather_data(weather_url)  
    if weather_df is not None:
        records = weather_df.to_dict(orient='records')
        i = 0
        for record in records:
            #producer.send('weather_test_topic_1', record)
            #producer.send('weather_test_topic_2', record)
            producer.send('weather_test_topic_45', record)
            if i < 1:
                print(record)
            i+=1
        #確保所有訊息都已發送
        producer.flush()            
        print("發送氣象成功 : ", weather_obs_time)
        
    else:
        print("Failed to get data from API.")
    
    
    '''
    for i in range(24): #
        #爬雨量、氣象資料
        weather_df, weather_obs_time = get_weather_data(weather_url)  
         
        if weather_df is not None:
            records = weather_df.to_dict(orient='records')
            i = 0
            for record in records:
                #producer.send('weather_test_topic_1', record)
                #producer.send('weather_test_topic_2', record)
                producer.send('weather_test_topic_3', record)
                    
            #確保所有訊息都已發送
            producer.flush()
            print("發送氣象成功 : ", weather_obs_time)
        else:
            print("Failed to get data from API.")

        time.sleep(60 * 60)  # 等待10分鐘
       
     
if __name__ == "__main__":
    main()
    
    
    