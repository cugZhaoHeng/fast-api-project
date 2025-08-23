import re
from datetime import date, datetime

def parse_date(text: str) -> date:
    """
    通用日期解析函数，支持多种中文和西文日期格式
    支持格式示例：
        '2016-1'       -> 2016-01-01
        '2016-11-23'   -> 2016-11-23
        '2016/8/5'     -> 2016-08-05
        '2016年8月'    -> 2016-08-01
        '2016年8月23日'-> 2016-08-23
        '2016'         -> 2016-01-01
        '出版于2015年'  -> 2015-01-01
    """
    if not text or not isinstance(text, str):
        return None

    # 清理文本
    text = text.strip()

    # 定义正则模式（支持多种分隔符和中文）
    patterns = [
        # 匹配：2016-08-23, 2016/08/23, 2016.08.23
        r'(\d{4})[^\d](0?[1-9]|1[0-2])[^\d](0?[1-9]|[12]\d|3[01])',
        # 匹配：2016-08, 2016/08, 2016.08
        r'(\d{4})[^\d](0?[1-9]|1[0-2])',
        # 匹配：2016年08月05日
        r'(\d{4})年(?:0?([1-9]|1[0-2]))月(?:0?([1-9]|[12]\d|3[01]))日?',
        # 匹配：2016年08月
        r'(\d{4})年(?:0?([1-9]|1[0-2]))月',
        # 匹配：2016年
        r'(\d{4})年',
        # 匹配：纯四位年份
        r'^(\d{4})$'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            year = int(groups[0])

            # 根据捕获组数量判断格式
            if len(groups) >= 3 and groups[2]:  # 年-月-日
                month = int(groups[1]) if groups[1] else 1
                day = int(groups[2])
            elif len(groups) >= 2 and groups[1]:  # 年-月（来自中文或分隔符）
                month = int(groups[1])
                day = 1
            else:  # 只有年
                month = 1
                day = 1

            # 验证日期合法性（防止 2016-02-30 这类错误）
            try:
                return date(year, month, day)
            except ValueError:
                try:
                    # 如果日非法（如 2月30日），设为当月最后一天或1号
                    return date(year, month, 1)
                except ValueError:
                    return None  # 年份或月份也不合法

    # 未匹配到任何格式
    print(f"⚠️ 无法解析日期字符串: {repr(text)}")
    return None