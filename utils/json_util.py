import json
from datetime import datetime, date

def serialize_obj(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def to_json(obj):
    return json.dumps(obj, default=serialize_obj, ensure_ascii=False)