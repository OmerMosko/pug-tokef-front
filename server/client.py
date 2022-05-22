import requests

url = "http://127.0.0.1:5000/uploader"

payload={}
files=[
  ('file',('file',open('./bla.txt','rb'),'application/octet-stream'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
