from kafka import KafkaProducer
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

#爬雨量資料
def get_rain_data(url):
    response = requests.get(url)
    
    # 抓資料
    if response.status_code == 200:
        data = response.json()
        
        # 提取 Station 列表中的資訊
        stations = data['records']['Station']
        
        # 創建一個空的 DataFrame
        df = pd.DataFrame(stations)
        
        # 將 GeoInfo 中的 Coordinates 提取出來，並展開為獨立的欄位
        df = pd.concat([df, df['ObsTime'].apply(lambda x: pd.Series(x))], axis=1)
        # 將 GeoInfo 中的 Coordinates 提取出來，並展開為獨立的欄位
        df = pd.concat([df, df['GeoInfo'].apply(lambda x: pd.Series(x['Coordinates'][0]))], axis=1)
        df = pd.concat([df, df['GeoInfo'].apply(lambda x: pd.Series(x))], axis=1)
        
        # 將 RainfallElement 中的資訊提取出來，並展開為獨立的欄位
        df = pd.concat([df, df['RainfallElement'].apply(lambda x: pd.Series(x))], axis=1)
        
        
        # 用'Precipitation'的值來替換'Past2days'和'Past3days'的值
        for col in df.columns[17:26]:
            df[col] = df[col].apply(lambda x: x['Precipitation'] if 'Precipitation' in x else 0.0)
        
        
        df['ObsTime'] = df['DateTime']
        # 刪除原始的 GeoInfo 和 RainfallElement 欄位
        df.drop(columns=['DateTime','GeoInfo', 'RainfallElement', 'Coordinates'], inplace=True)
        
        #取得obs_time並處理字串
        obs_time = df.at[0,'ObsTime'][:19].replace('T', ' ')
        
        df['ObsTime'] = obs_time
        
        print(obs_time)
        
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
    #爬雨量、氣象資料 #, rain_obs_time
    rain_df, rain_obs_time = get_rain_data(rain_url)

    if rain_df is not None:   
        #將DataFrame轉為字典，並發送到Kafka
        records = rain_df.to_dict(orient='records')
        i = 0
        for record in records:
            #producer.send('rain_test_topic_1', record)
            #producer.send('rain_test_topic_2', record)
            #producer.send('rain_test_topic_45', record)
            if i < 1:
                print(record)
            i+=1

        #確保所有訊息都已發送
        #producer.flush()
        print("發送雨量成功 : ", rain_obs_time)
    else:
        print("Failed to get data from API.")
    
    '''   
    #持續抓的迴圈===================================
    for i in range(144):
        #爬雨量、氣象資料
        rain_df, rain_obs_time = get_rain_data(rain_url)
        
        if rain_df is not None:   
            #將DataFrame轉為字典，並發送到Kafka
            records = rain_df.to_dict(orient='records')
            for record in records:
                #producer.send('rain_test_topic_1', record)
                #producer.send('rain_test_topic_2', record)
                producer.send('rain_test_topic_3', record)
                    
            #確保所有訊息都已發送
            producer.flush()
            print("發送雨量成功 : ", rain_obs_time)
        else:
            print("Failed to get data from API.")

        time.sleep(10 * 60)  # 等待10分鐘
       
if __name__ == "__main__":
    main()