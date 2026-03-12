import requests

url = "http://127.0.0.1:8989/file_parse"  # 根据你的实际路径调整
file_path = "/home/tlw/下载/软件栈安装指南.pdf"

with open(file_path, 'rb') as f:
    files = {'files': (file_path, f, 'application/pdf')}
    data = {'backend': 'hybrid-auto-engine'}
    try:
        response = requests.post(url, files=files, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")