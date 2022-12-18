# Rock-paper-scissors image classification (artificial dataset)
This was made as the Capstone 1 project for the Machine Learning Engineering class ([mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp)). Therefore, the model or the application itself are not particularly useful, rather it is a learning effort for the process of exploring the process behind creating and (locally) deploying a custom classification model using Kubernetes/KServe.

## Potential usage

Can be adapted to be used as an automated referee and score keeping for rock-paper-scissors games.

## Current status

For this project, a CNN-based neural network classification model was developed.
The model itself takes a grayscale image (150,150) of a hand as input.
The deployed model takes an URL of an image. 

The model predicts the probability of the image containing a hand in one of the three poses:
 - rock
 - paper
 - scissors

## Dataset 
Download the dataset from [Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset). Make sure to have only a set of folders (not multiple versions of the same dataset) in the rock\_paper\_scissors\_dataset folder with the train, validation and test structure.

  ![Dataset](https://github.com/daniel-gheorghita/mlzoomcamp/blob/main/Projects/Capstone_1_rock_paper_scissor_classification/sample_dataset.png)

## Model development
Install [Anaconda](https://www.anaconda.com/products/distribution).

Use the environment.yml file to create a Conda environment: 
```sh
conda env create -f environment.yml
```

Then activate it (use the new name, in case you changed it):
```sh
conda activate mlzoomcamp
```

Use the rock\_paper\_scissors.ipynb for data exploration and experimenting on various models. 

After running the experimentation section of the notebook, use the train.py script to train the final selected model (edit the desired model parameters based on the experiments' output). 
```sh
python3 train.py
```

The script exports a  [Tensorflow SavedModel](https://www.tensorflow.org/guide/saved_model) model (and checkpoints in the h5 format). The model folder needs to be named by a version number, packed in a zip archive and uploaded to an accessible URI (or via a local http server). There already is a model available on an [AWS S3 bucket](https://mlzoomcamp.s3.eu-central-1.amazonaws.com/rock_paper_scissors_model/rps-model-1.zip).

Hint: before archiving, rename the folder containing the model to "1". Otherwise, the service URLs in "predict.py" and in the sample scripts below need to be changed to match the version folder name.

## Deployment (local)
Install [Docker](https://www.docker.com/).

[KServe](https://kserve.github.io/website/0.9/get_started/) is used for deployment (requires [kind](https://kind.sigs.k8s.io/docs/user/quick-start) and [Kubernetes CLI](https://kubernetes.io/docs/tasks/tools/)).

Create the kind cluster
```sh
kind create cluster
```

Start the model inference service (it uses the tensorflow.yaml file):
```sh
kubectl apply -f tensorflow.yaml
```

Check if the inference service and the transformer are running:
```sh
kubectl get pod
```

The output should be a list containing:
 - rock-paper-scissors-predictor-default-<number>-deployment-<ID>
 - rock-paper-scissors-transformer-default-<number>-deployment-<ID>

And 2/2 under the "READY" column for both. In case the "READY" column does not display 2/2, one quick way to debug is to use the command (this is also helpful to confirm the correct model version and the port on which it is exposed):
```sh
kubectl logs rock-paper-scissors-predictor-default-<number>-deployment-<ID> kserve-container
```

or 

```sh
kubectl logs rock-paper-scissors-transformer-default-<number>-deployment-<ID> kserve-container
```

One common reason for failing to run these Kubernetes pods is insufficient compute power.

Explanation: 
 - the predictor is created by KServe by using [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) with the custom model zip file (mentioned above);
 - the transformer is a helper mechanism that takes an URL and extracts the image, transforms it to grayscale and scales the pixel values to [0,1]; the implementation and Dockerfile can be found under the /image_transformer folder and is already available on the [Docker Hub](https://hub.docker.com/layers/danielghe/mlzoomcamp/rps\_transformer/images/sha256-d54d25b39705e2e2bbd7aae5b77e347dbe1b54b7fd15624f6ba41c8a01a4d48a?context=explore). 

Before testing the prediction service, make sure to forward the port from the local machine to the service (needs to run continuously):

```sh
kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80
```

Now the prediction can be tested:
```sh
python3 predict.py
```

## Prediction service development/customization hints

The prediction service can be tested without a transformer by removing the "transformer" fields from the tensorflow.yaml file. 

Then, the command
```sh
kubectl apply -f tensorflow.yaml
```
should only display a running service (the prediction).

The prediction service can be tested independently by a simple Python script (that is very similar to what the image_transformer actually implements):
```python
import requests
from PIL import Image
import numpy as np
# open method used to open different extension image file
im = Image.open(r"local_image.png").resize((150,150)).convert('L')
X = np.array([np.expand_dims(np.array(im, dtype=np.float32),axis=-1)]) / 255
service_name = 'rock-paper-scissors'
host = f'{service_name}.default.example.com'
actual_domain = 'http://localhost:8080'
url = f'{actual_domain}/v1/models/{service_name}:predict'
headers = {'Host': host}
print(host, actual_domain, url, headers)
# kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80   <- has to be running
request = {'instances' : X.tolist()}
response = requests.post(url, json=request, headers=headers).json()
print(response)
```

More detailes can be found on the KServe [serving Tensorflow model](https://kserve.github.io/website/modelserving/v1beta1/tensorflow/) tutorial (and how to replace the script with a curl call).

## Transformer service development/customization hints

Running a transformer without publishing the image can be done:
```sh
python3 image_transformer.py --predictor_host=localhost:8080 --model_name=rock-paper-scissors --http_port=8081
```

Add the port-forwarding (the previous port-forwarding still needs to be running):
```sh
kubectl port-forward rock-paper-scissors-predictor-default-<number>-deployment-<ID> 8080:8080
```

In case of error "error: unable to listen on any of the requested ports: [{8080 8080}]" use the bellow instructions to terminate the port binding:
```sh
lsof -i :8080
kill -9 <pid>
```
This error appears because kubectl does not unbind ports.

Sample Python script for testing the image-transformer service locally (can be replaced with an appropiate curl call).
```python
from PIL import Image
import requests
from io import BytesIO
url_img = "https://mlzoomcamp.s3.eu-central-1.amazonaws.com/rock_paper_scissors_model/test/rock/testrock01-00.png"
response = requests.get(url_img)
img = Image.open(BytesIO(response.content))
service_name = 'rock-paper-scissors'
host = f'{service_name}.default.example.com'
actual_domain = 'http://localhost:8081' # for local deployment of the image transformer
url = f'{actual_domain}/v1/models/{service_name}:predict'
headers = {'Host': host}
request = {'instances' : [url_img]}
response = requests.post(url, json=request, headers=headers).json()
```

More detailes can be found on the KServe [Transformer](https://kserve.github.io/website/modelserving/v1beta1/transformer/torchserve_image_transformer/) tutorial.

If the behavior is acceptable, one option is to push the transformer image to Docker Hub (make sure you have an account and a repository already created):
```sh
docker login
docker build . -t rps_transformer
docker tag rps_transformer:latest <username>/<repository>:rps_transformer
docker push <username>/<repository>:rps_transformer
```

## Useful kubectl commands

Short cheatsheet:
 - create cluster: kind create cluster
 - create a namespace: kubectl create namespace namespace_name
 - create pod/service: kubectl apply -f tensorflow.yaml -n namespace_name
 - check the IP and ports of the inference server: kubectl get svc istio-ingressgateway -n istio-system
 - forward the local port 8080 to port 80 of the inference server: kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80
 - list the available pods (default namespace): kubectl get pod
 - list the ports in a given namespace: kubectl get pod -n namespace_name
 - list the available namespaces: kubectl get namespace
 - get inference service host name: kubectl get inferenceservices rock-paper-scissors
 - get inference service host name in a namespace: kubectl get inferenceservices rock-paper-scissors -n namespace_name
 - logs of the container running the model: kubectl logs <pod_name> kserve-container
 - logs of the container running the model: kubectl logs <pod_name>


