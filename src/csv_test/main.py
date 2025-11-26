import json

import pandas as pd

df = pd.read_csv('a.csv', delimiter=",")
df = df.astype(object)
df = df.where(pd.notna(df), None)
print(df)
a = df.to_dict(orient='records')
print(a)
print(json.dumps(obj=a, ensure_ascii=False))
print("============")

b = df.to_json(orient='records')
print(b)

a[0]['c'] = b
print(a)
c = print(json.dumps(a))
print(c)

d = {
    "a": 1,
    "b": None
}
print(d)
print(json.dumps(d))