from datetime import datetime
import time
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt   #繪圖
from sklearn.preprocessing import LabelEncoder      #linReg 
from sklearn.linear_model import LinearRegression   #linReg

#===LSTM======
import pandas as pd
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#爬資料
def get_data_from_api(url):
    response = requests.get(url)
    
    # 抓資料
    if response.status_code == 200:
        data = response.json()
        
        # 遍歷每個地點的資料
        for location in data['records']['location']:
            weather_elements = {}
            
            # 提取 weatherElement 列表中的資訊
            for element in location['weatherElement']:
                key = element['elementName']
                value = element['elementValue']
                weather_elements[key] = value
            
            # 提取 parameter 列表中的資訊
            parameters = {}
            for param in location['parameter']:
                key = param['parameterName']
                value = param['parameterValue']
                parameters[key] = value

            # 更新地點資料，並刪除原始的 weatherElement 和 parameter 列表
            location.update(weather_elements)
            location.update(parameters)
            del location['weatherElement']
            del location['parameter']
        
        df = pd.json_normalize(data['records']['location'])
        
        # 取得時間
        obs_time = data['records']['location'][0]['time']['obsTime']
        return df, obs_time
    else:
        print(f"Error: {response.status_code}")
        return None, None

# 將資料存到資料夾
def save_to_csv(df, obs_time, base_folder_name):
    # 檢查 DataFrame 是否有效
    if df is not None and isinstance(df, pd.DataFrame):

        # 從觀察時間中獲取日期和時間
        date_str = obs_time.split(" ")[0].replace("-", "")
        time_str = obs_time.split(" ")[1].replace(":", "")
        # 格式化時間以用於檔名
        formatted_time = obs_time.replace(":", "").replace("-", "").replace(" ", "_")  #資料內的時間
        
        # 創建基於日期的資料夾路徑
        date_folder_path = os.path.join(base_folder_name, date_str)

        # 檢查資料夾是否存在，若不存在則創建資料夾
        if not os.path.exists(date_folder_path):
            os.makedirs(date_folder_path)
        
        # 儲存 DataFrame 到 CSV
        output_path = os.path.join(date_folder_path, f"output_data_{formatted_time}.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Data saved to {output_path}")
    else:
        print("Data is not in DataFrame format")

#到資料夾提取csv全部資料出來(暫時不要用，全部抓的話資料太多了，用get_location_data可抓特定資料)
def get_all_data_from_csv(base_directory):
    result = []

    for date_folder in os.listdir(base_directory):
        date_folder_path = os.path.join(base_directory, date_folder)
        
        if os.path.isdir(date_folder_path):
            for filename in os.listdir(date_folder_path):
                if filename.endswith('.csv'):
                    filepath = os.path.join(date_folder_path, filename)
                    df = pd.read_csv(filepath, encoding='utf-8-sig')
                    
                    # 如果數值小於 0 則設為 0
                    df[df.select_dtypes(include=[np.number]) < 0] = 0
                    
                    # 將提取的資料加入結果列表中
                    result.append(df)

    result_df = pd.concat(result, ignore_index=True)
    return result_df

# 從資料夾提取特定地點的特定欄位資料
def get_location_data(location_names, column_names, base_directory):
    result_data = {name: {col: [] for col in column_names} for name in location_names}
    
    for date_folder in os.listdir(base_directory):
        date_folder_path = os.path.join(base_directory, date_folder)
        
        if os.path.isdir(date_folder_path):
            for filename in os.listdir(date_folder_path):
                if filename.endswith('.csv'):
                    filepath = os.path.join(date_folder_path, filename)
                    df = pd.read_csv(filepath, encoding='utf-8-sig')

                    # 如果數值小於 0 則設為 0
                    df[df.select_dtypes(include=[np.number]) < 0] = 0

                    for location in location_names:
                        for col in column_names:
                            if location in df['locationName'].values:
                                value = df[df['locationName'] == location][col].values[0]
                                result_data[location][col].append(value)
                            else:
                                result_data[location][col].append(None)
    
    result_df = {}
    for location, columns_data in result_data.items():
        result_df[location] = pd.DataFrame(columns_data)

    return result_df

# 從資料夾提取特定地點的特定欄位資料，以欄位當作index
def get_location_data_indexTime(location_names, column_names, base_directory):
    result_data = {name: {col: [] for col in column_names} for name in location_names}
    time_stamps = []
    
    for date_folder in os.listdir(base_directory):
        date_folder_path = os.path.join(base_directory, date_folder)
        
        if os.path.isdir(date_folder_path):
            for filename in os.listdir(date_folder_path):
                if filename.endswith('.csv'):
                    timestamp = filename.replace("output_data_", "").replace(".csv", "").replace("_", " ")
                    time_stamps.append(timestamp)

                    filepath = os.path.join(date_folder_path, filename)
                    df = pd.read_csv(filepath, encoding='utf-8-sig')

                    # 如果數值小於 0 則設為 0
                    df[df.select_dtypes(include=[np.number]) < 0] = 0

                    for location in location_names:
                        for col in column_names:
                            if location in df['locationName'].values:
                                value = df[df['locationName'] == location][col].values[0]
                                result_data[location][col].append(value)
                            else:
                                result_data[location][col].append(None)
    
    result_df = {}
    for location, columns_data in result_data.items():
        temp_df = pd.DataFrame(columns_data)
        temp_df['Timestamp'] = time_stamps
        temp_df.set_index('Timestamp', inplace=True)
        result_df[location] = temp_df

    return result_df

#畫折線圖
def plot_data(data, column_names, start_time, end_time):
    color_list = ["red", "blue", "orange", "green", "purple", "brown", "pink"]

    for location, df in data.items():
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.figure(figsize=(12, 6))

        for i in range(len(column_names)):
            plt.plot(df[column_names[i]], label=column_names[i], color=color_list[i])

            plt.title(f"{location} {start_time}~{end_time}")
            plt.xlabel("時間")
            plt.ylabel("數值")
            plt.legend()
    
            # 選擇每18組的刻度來顯示
            selected_ticks = df.index[::18]
            plt.xticks(selected_ticks, rotation=80)
    
            plt.show()

#LinearRegression預測
def predict_linReg(df):
    """
    需給定一個DataFrame
    地點名稱作為X，而特徵（如'NOW'）作為y。
    """
    
    # 對地點名稱進行編碼
    label_encoder = LabelEncoder()
    encoded_location = label_encoder.fit_transform(df.columns)

    predictions = {} #放每個地點及其預測到的值

        
    for location, encoded_loc in zip(df.columns, encoded_location):  #(地名, 地名編碼)
        X = [[encoded_loc]] * len(df)   #地名編碼*幾筆資料
        y = df[location].values         #地名的所有雨量資料
        
        
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict([[encoded_loc]])

        predictions[location] = pred[0]       #{地點:預測到的值}


    # 顯示預測結果
    print("預測結果 : ")
    for location, pred_value in predictions.items():   #印出來{地點:預測到的值}
        print(f"{location}: {pred_value:.2f}")   

    return predictions

#找出'雨量'、'氣候'有重複的氣象站       
def find_repeat_locations(directory1, directory2):
    repeat_locations = set()

    # 抓出"氣候"資料夾內的觀測站
    for date_folder in os.listdir(directory1):
        date_folder_path = os.path.join(directory1, date_folder)
        if os.path.isdir(date_folder_path):
            for filename in os.listdir(date_folder_path):
                if filename.endswith('.csv'):
                    filepath = os.path.join(date_folder_path, filename)
                    df = pd.read_csv(filepath)
                    repeat_locations.update(df['locationName'].values)

    # 找"雨量"資料夾內有重複的觀測站
    for date_folder in os.listdir(directory2):
        date_folder_path = os.path.join(directory2, date_folder)
        if os.path.isdir(date_folder_path):
            for filename in os.listdir(date_folder_path):
                if filename.endswith('.csv'):
                    filepath = os.path.join(date_folder_path, filename)
                    df = pd.read_csv(filepath)
                    locations = set(df['locationName'].values)
                    repeat_locations = repeat_locations.intersection(locations)

    print(list(repeat_locations))  #印出重複的觀測站
                                  
#將'氣象'與'雨量'合併
def merge_data_from_weather(df, location_names, column_names, base_directory="氣象觀測資料輸出"):
    weather_data = {location: {col: [] for col in column_names} for location in location_names}

    def parse_datetime_from_filename(filename):
        date, time = filename.split('_')[2:4]
        full_time = f"{date} {time[:2]}:{time[2:4]}:00"
        return full_time

    for date_folder in os.listdir(base_directory):
        date_folder_path = os.path.join(base_directory, date_folder)

        if os.path.isdir(date_folder_path):
            for filename in os.listdir(date_folder_path):
                if filename.endswith('.csv'):
                    filepath = os.path.join(date_folder_path, filename)
                    meteo_df = pd.read_csv(filepath, encoding='utf-8-sig')
                    timestamp = parse_datetime_from_filename(filename)
                    
                    # 遍歷df中的每小時資料
                    for idx in range(0, 60, 10):
                        time_idx = (pd.Timestamp(timestamp) + pd.Timedelta(minutes=idx)).strftime('%Y%m%d %H%M%S')
                        
                        if time_idx in df[location_names[0]].index:  # Check for one location, assuming all locations have the same index
                            for location in location_names:
                                for col in column_names:
                                    value = meteo_df[meteo_df['locationName'] == location][col].values
                                    if value:
                                        weather_data[location][col].append(value[0])
                                    else:
                                        weather_data[location][col].append(np.nan)

    result_dfs = {location: pd.DataFrame(weather_data[location], index=df[location].index).fillna(0) for location in location_names}
    for location in location_names:
        result_dfs[location] = result_dfs[location].clip(lower=0) 
    
    merged_data = {}
    for location in df.keys():
        merged_data[location] = pd.concat([df[location], result_dfs[location]], axis=1).fillna(0)
    
    return merged_data

#LSTM預測
def predict_lstm(df, location_names, columns, features_to_train, look_back, target_column):
    #正規化
    def normalize(df, columns):
        numeric_columns = df[columns].astype(float)
        #numeric_columns = pd.DataFrame(numeric_columns)
        minimum = numeric_columns.min()
        maximum = numeric_columns.max()
        norm = (numeric_columns - minimum) / (maximum - minimum)
        norm.fillna(0, inplace=True)
        return norm, maximum, minimum
    
    # 反正規化函數
    def denormalize(norm_data, max_val, min_val):
        return [n * (max_val - min_val) + min_val for n in norm_data]
    
    
    #依照要往前參考的資料筆數儲存train資料
    def train_windows(df, features, target_column, look_back, predict_day=3):
        X_train, Y_train, timestamps = [], [], []
        for i in range(df.shape[0] - predict_day - look_back):
            # 使用指定的 features 來選擇數據
            X_train.append(np.array(df[features].iloc[i: i + look_back]))
            Y_train.append(np.array(df.iloc[i + look_back: i + look_back + predict_day][target_column]))
            timestamps.append(df.index[i + look_back])
        return np.array(X_train), np.array(Y_train), timestamps
    
    #模型
    def lstm_stock_model(shape):
        model = Sequential()
        model.add(LSTM(256, input_shape=(6, len(features_to_train)), return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.add(Flatten())
        model.add(Dense(5,activation='linear'))
        model.add(Dense(1,activation='linear'))
        model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
        model.summary()
        return model
    
    #繪製預測與測試資料
    def plot_predictions(test_timestamps, Y_test, pred, location, target_column):
        plt.figure(figsize=(15, 6))
        plt.plot(test_timestamps, Y_test, color='red', label='實際雨量') 
        plt.plot(test_timestamps, pred, color='blue', label='預測雨量')  # 使用test_timestamps作為x軸
        plt.title(f'{location}[{target_column}]預測值 vs 實際值')
        plt.xlabel('時間')
        # 設置X軸的刻度
        plt.xticks(test_timestamps[::18], rotation=70)  # 使用slicing每18組取一個刻度，rotation字旋轉幾度
        plt.ylabel('雨量')
        plt.legend()
        # 顯示主要的格線
        plt.grid(True, which='major', linestyle='-', linewidth=0.5)
        # 打開次要的刻度
        plt.minorticks_on()
        # 顯示次要的格線，但使其更淡和更小
        #plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
        plt.tight_layout()  # 調整佈局
        plt.show()
    
    #繪製loss   
    def plot_loss(history, location):
        fig, ax = plt.subplots()  # 使用subplots來獲得ax物件
        
        ax.plot(history.history['loss'], color='royalblue', linestyle="-")
        ax.plot(history.history['val_loss'], color='orange', linestyle="-")
        ax.legend(['loss', 'val_loss'])
        ax.set_title(f"{location} loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    
        # 主要格線
        ax.grid(True, which='major', linestyle='-', linewidth=0.5)
        # 次要格線
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    
        plt.show()
    

    
    for location in location_names:  #多個地點
        location_df = df[location] 
        
        #location_df, max_val, min_val = normalize(df[location], columns)

        # 切分為訓練和測試集
        train_size = int(len(location_df) * 0.8)
        train, test = location_df[0:train_size], location_df[train_size:]
        
        #train_windows
        X_train, Y_train, _ = train_windows(train, features_to_train, target_column, look_back, 1)
        X_test, Y_test, test_timestamps = train_windows(test, features_to_train, target_column, look_back, 1)

        model = lstm_stock_model(X_train.shape)
        
        #mean_absolute_error在連續patience訓練週期中不再改進時，訓練將被終止
        callback = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")  
        
        #訓練一個神經網路模型
        history = model.fit(X_train, Y_train,  epochs=1000, batch_size=len(X_train),validation_split=0.1, callbacks=[callback],shuffle=True)
        #\儲存模型
        model.save('LSTM_model.h5')
        
        #預測
        pred = model.predict(X_test)
        
        # 在預測後將所有負數轉為0
        pred[pred < 0] = 0
        
        # 在模型預測後進行反正規化
        #pred = denormalize(pred.flatten(), max_val, min_val)
        #Y_test = denormalize(Y_test, max_val, min_val)
        
        #評估模型
        rmse = np.sqrt(mean_squared_error(Y_test, pred))
        print("RMSE : ", rmse)
      
        
        
        #繪製預測與測試資料
        plot_predictions(test_timestamps, Y_test, pred, location, target_column)
        #繪製loss
        plot_loss(history, location)




def main():
    '''
    url = "https://opendata.cwb.gov.tw/api/v1/rest/datastore/O-A0002-001?Authorization=CWB-362C1548-B175-4BE4-AEDD-3978655ED307"

    df, obs_time = get_data_from_api(url)   #爬資料
    save_to_csv(df, obs_time)               #將資料存到資料夾
    '''
    
    
    
    location_names = '外埔'.split(",")      #抓哪些地點(可以自己改)
    rain_column_names = 'HOUR_12'.split(",")            #抓雨量哪個欄位(可以自己改)
    
    df = get_location_data_indexTime(location_names, rain_column_names, "雨量觀測資料輸出")   #到資料夾提取csv特定地點特定資料
    #plot_data(df, rain_column_names, '20230908 12:00', '20230911 08:50')   #畫折線圖
    #print(df, "\n")
    
    weather_column_names = 'WDSD,TEMP,HUMD,PRES'.split(",")     # 抓氣象哪個欄位
    
    merge_df = merge_data_from_weather(df, location_names, weather_column_names)
    #plot_data(merge_df, rain_column_names+weather_column_names, '20230908 12:00', '20230911 08:50')   #畫折線圖
    
    #將欄位名稱更改並將 'RelativeHumidity' 的值乘以 100
    column_mapping = {
    'HOUR_12': 'Past12hr',
    'WDSD': 'WindSpeed',
    'TEMP': 'AirTemperature',
    'HUMD': 'RelativeHumidity',
    'PRES': 'AirPressure',
    }
    merge_df = {location: data.rename(columns=column_mapping).apply(lambda x: x * 100 if x.name == 'RelativeHumidity' else x) for location, data in merge_df.items()}
    
    for location, merge in merge_df.items():
        print(f"------ {location} ------")
        print(merge)
        print("\n")  
    
    features_reserve = ['Past12hr','WindSpeed','AirTemperature','RelativeHumidity','AirPressure']
    features_to_train = ['WindSpeed','AirTemperature','RelativeHumidity','AirPressure']  #要用甚麼欄位訓練
    predict_lstm(merge_df, location_names, features_reserve, features_to_train, look_back=6, target_column='Past12hr') #LSTM預測
    
    #predict_linReg(df)    #LinearRegression預測
    
    #find_repeat_locations(directory1='氣象觀測資料輸出', directory2='雨量觀測資料輸出')  #找出'雨量'、'氣候'有重複的氣象站   

    

#持續抓資料的迴圈
'''
    total_intervals = 144   #抓幾小時 (60/每隔幾分鐘 * 小時)

    for i in range(total_intervals):
        print("抓取時間 : ", datetime.now())     #印抓取的時間
        df, obs_time = get_data_from_api(url)   #抓資料
        save_to_csv(df, obs_time, '雨量觀測資料輸出')               #存檔
        #print(df)   
        time.sleep(10 * 60)  # 等待10分鐘
'''

   
if __name__ == "__main__":
    main()