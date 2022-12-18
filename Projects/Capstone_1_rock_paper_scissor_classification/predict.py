import requests
from PIL import Image
import numpy as np

# URL test files
url_img = []
url_img.append("https://mlzoomcamp.s3.eu-central-1.amazonaws.com/rock_paper_scissors_model/test/rock/testrock01-00.png")
url_img.append("https://mlzoomcamp.s3.eu-central-1.amazonaws.com/rock_paper_scissors_model/test/scissors/testscissors01-00.png")

# Sample inference for local deployment
# kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80   <- has to be running locally
service_name = 'rock-paper-scissors'
host = f'{service_name}.default.example.com'
actual_domain = 'http://localhost:8080'
#actual_domain = 'http://localhost:8081' # for local deployment of the image transformer
url = f'{actual_domain}/v1/models/{service_name}:predict'
headers = {'Host': host}
print(host, actual_domain, url, headers)

request = {'instances' : url_img}
response = requests.post(url, json=request, headers=headers).json()

print(response)