#!/usr/bin/env python3
import json
import sys


def check_pressure(json_file):
    # 范围标准
    MIN_VAL = 2.5
    MAX_VAL = 11.8

    # 统计
    abnormal_records = []
    total_checked = 0

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {json_file} 不存在")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file} 不是有效的JSON格式")
        return

    # 遍历检测
    for item in data.get('stationPredictDTOS', []):
        station_id = item.get('stationId')
        output = item.get('outputPress')
        input_val = item.get('inputPress')
        time_val = item.get('time')

        # 检查outputPress
        if output is not None and (output < MIN_VAL or output > MAX_VAL):
            abnormal_records.append({
                'stationId': station_id,
                'field': 'outputPress',
                'value': output,
                'time': time_val
            })

        # 检查inputPress
        if input_val is not None and (input_val < MIN_VAL or input_val > MAX_VAL):
            abnormal_records.append({
                'stationId': station_id,
                'field': 'inputPress',
                'value': input_val,
                'time': time_val
            })

        total_checked += 1

    # 输出结果
    print(f"检查完成！")
    print(f"共检查 {total_checked} 条记录")
    print(f"发现 {len(abnormal_records)} 个异常值")
    print("-" * 60)

    if abnormal_records:
        # 按stationId统计
        station_stats = {}
        for rec in abnormal_records:
            sid = rec['stationId']
            if sid not in station_stats:
                station_stats[sid] = 0
            station_stats[sid] += 1

        print("按站点统计：")
        for sid, count in station_stats.items():
            print(f"  stationId: {sid}, 异常数: {count}")

        print("\n详细异常记录：")
        for rec in abnormal_records:
            print(f"  stationId: {rec['stationId']}, "
                  f"字段: {rec['field']}, "
                  f"值: {rec['value']}, "
                  f"时间戳: {rec['time']}")
    else:
        print("✅ 所有数据都在正常范围内 (2.5 ~ 11.8)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_pressure.py <json文件路径>")
        sys.exit(1)

    check_pressure(sys.argv[1])