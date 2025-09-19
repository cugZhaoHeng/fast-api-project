from datetime import datetime

import pandas as pd
from requests import request


# 将时间转换为整点并转为时间戳
def convert_to_timestamp(dt_str):
    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    hour_start = dt.replace(minute=0, second=0, microsecond=0)
    return int(hour_start.timestamp())

if __name__ == '__main__':
    with open(file="./pressure.xlsx", mode="rb") as f:
        pd_data = pd.read_excel(io=f)
        columns_list : list[str] = pd_data.columns.tolist()
        df_renamed = pd_data.rename(columns={
            columns_list[0]: 'stationId',
            columns_list[1]: 'inputPress',
            columns_list[2]: 'outputPress',
            columns_list[3]: 'time'
        })


        # 将json数据的时间转化为时间戳（整点）
        df_renamed['time'] = df_renamed['time'].apply(convert_to_timestamp)
        # 按照行来转换为JSON
        json_records = df_renamed.to_json(orient="records")
        print(json_records)
        # 调用接口发送数据
        url = "http://192.168.110.35:30046/baidu/pipeline/api/station/receive_station_predict_data"
        headers = {
            'accept': '*/*',
            'Authorization': 'auth',
            'Tetproj': '05c911b5efbc4f53a659ac27fbc16614',
            'Content-Type': 'application/json'
        }
        print("\n正在发送数据到接口...")
        response = request("POST", url, headers=headers, data=json_records)
        print("返回码：" + str(response.status_code))
        if response.status_code == 200:
            print("数据发送成功！")
        else:
            print("数据发送失败，返回信息：" + response.text)