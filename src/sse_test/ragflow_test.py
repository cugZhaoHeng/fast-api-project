import requests
import json

url = "http://192.168.111.139/v1/conversation/ask"

payload = json.dumps({
  "kb_ids": [
    "163a261511ef11f1818bae41e2bdf18c"
  ],
  "question": "气相相对渗透率",
  "tenantId": None,
  "search_id": "54c530cf120311f1a7ef268fa276b933"
})
headers = {
  'Accept': '*/*',
  'Accept-Language': 'zh-CN,zh;q=0.9',
  'Authorization': 'ImQwZjUyMWFlMTJlYTExZjE4MGFlMmE2MWQ4NDgxYjEzIg.aaAATA.Hi-hwSUnOpfzhe4dmD7xHVHpKBo',
  'Connection': 'keep-alive',
  'Content-Type': 'application/json',
  'Origin': 'http://192.168.111.139',
  'Referer': 'http://192.168.111.139/next-search/54c530cf120311f1a7ef268fa276b933?page=1',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
  'Cookie': 'session=.eJwdyzsSgCAMBcC7pLYgRJBwGQaEN9qiVI5391NusRelcbSe9kqRHByygRrvwAzOCizNWJSZZa1MEyX0dmwUzz7aq7_NBUFVmK2Rr5VQs_UB2S6-qAjdDyBTHUs.aZ573w.Hbcl_xgGIpGCoCJcN_Oc6vxoz6A'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
