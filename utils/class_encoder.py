import json
from datetime import date, datetime

class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()  # 转为 "2025-08-23"
        if isinstance(obj, datetime):
            return obj.isoformat()  # 转为 "2025-08-23T10:00:00"
        return super().default(obj)