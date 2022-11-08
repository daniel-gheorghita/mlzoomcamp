# Song genre prediction based on audio features (Spotify dataset)
This was made as a midterm project for the Machine Learning Engineering class ([mlzoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp)). Therefore, the model or the application itself are not particularly useful, rather it is a learning effort for the process of exploring multiple models and deploying on the cloud. Having used AWS and GCP in the past, I wanted to make a point of deploying this project on Azure as a first contact with the Microsoft platform. 

## Potential usage

Can be used as a rough recommender system that suggests a class of music genres based on desired audio-mood features (danceability/energy).

## Current status

For this project, a binary classification model was developed.
The model has two continuous input parameters:
- danceability [0, 1]
- energy [0, 1]

Based on these features, the model predicts if the song is in one of the two classes:
 - rock/alternative/metal (output 1)
 - blues/hiphop/indie (output 0)

## Dataset 
Use the dataset already in the raw\_data folder or download an updated version from [spotify-multigenre-playlists-data](https://www.kaggle.com/datasets/siropo/spotify-multigenre-playlists-data). Make sure to have only a set of CSV files (not multiple versions of the same dataset) in the raw\_data folder because the first cells of notebook.ipynb will concatenate them all into a single file (spotify_songs.csv) that will be stored in the data folder. Then another file will be created (spotify_songs_small.csv) with only a subset of the columns from the original file. Only this final output data file is used for training the model.   

## Development
Install [Anaconda](https://www.anaconda.com/products/distribution).

Use the environment.yml file to create a Conda environment: 
```sh
conda env create -f environment.yml
```

Then activate it (use the new name, in case you changed it):
```sh
conda activate mlzoomcamp
```

Use the notebook.ipynb for data exploration, cleaning and experimenting on various models. 

Use the train.py script (or the train.ipynb notebook) to train the final selected model. 
```sh
python3 train.py
```

The script exports a model using [BentoML](https://www.bentoml.com/). The model is then used in the service.py to be exposed as an API. 

## Simple local serving (API available on http://0.0.0.0:3000 via Swagger UI) 
```sh
bentoml serve service:svc
```

## Deployment
Install [Docker](https://www.docker.com/).

[BentoML](https://www.bentoml.com/) is also used for facilitating deployment. 

Build the bento model (it uses the bentofile.yaml file):
```sh
bentoml build
```

Generate the Docker container (adjust model name and tag based on the output of the previous command):
```sh
bentoml containerize song_classifier:o4dx2rc2dsv6tjgr --platform linux/amd64
```

The generated contents (including the Dockerfile) for this particular model are available in the bentos/song_classifier/o4dx2rc2dsv6tjgr/env/docker folder. 

### Locally

Serve the model locally (adjust the name and tag of the image based on the previous command):
```sh
docker run -it --rm -p 3000:3000 song_classifier:o4dx2rc2dsv6tjgr serve --production
```
Access localhost:3000 or 0.0.0.0:3000 to access the API in the browser.

### Cloud (Azure)
A model was deployed as App Service in Azure using Docker: [https://songgenre2.azurewebsites.net/](https://songgenre2.azurewebsites.net/)

(It will not be active forever, so don't be surprised if it does not work anymore.)

Install Azure-CLI (via pip, Conda packages are not well maintained):
```sh
pip install azure-cli
```
Open the login screen (a webpage will appear in your browser for login):
```sh
az login
```

Login on the Azure Container Registry (create one first in the Azure web console - I used the name songgenre, adapt the command for the name used in your setup):
```sh
az acr login -n songgenre
```

Change the tag of the docker image by adding the ACR name in front of it (use docker image ls for all the images available on your system to adjust this command based on the image name and tag for your setup):
```sh
docker image tag 0ef8c7d239b2 songgenre.azurecr.io/song_classifier:o4dx2rc2dsv6tjgr
```

Push the image to ACR (adjust the name and tag for your setup):
```sh
docker push songgenre.azurecr.io/song_classifier:o4dx2rc2dsv6tjgr
```

Enable admin rights for resources in order to be able to deploy it as an App Service:
```sh
az acr update -n songgenre --admin-enabled true
```

When creating the App Service using the Docker container, one needs to specify the startup command as 
```sh
serve --production
```

In order to expose the port 3000 (or any port of your choice), one needs to add to Application Settings the pair WEBSITE_PORTS=3000. This can be done via the web console or via the command line:
```sh
az webapp config appsettings set --name <app-name> --resource-group <group-name> --settings WEBSITE_PORTS="3000"
```

### Verify deployment 
In order to check that your deployment works, either go to the localhost:3000 or the cloud API and use the Swagger UI to send POST requests.

Alternatively, one can use the curl command (as exemplified at the end of the train.ipynb) for testing the local deployment or the cloud deployment respectively:
```sh
curl -X 'POST' 'http://0.0.0.0:3000/classify_song' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"danceability":0.5, "energy":0.5}'
```
```sh
curl -m 300 -X 'POST' 'https://songgenre2.azurewebsites.net/classify_song' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"danceability":0.5, "energy":0.5}'
```

# Final word
I initially aimed at finding a model that would predict song popularity based on the audio features. It did not work out well as the feature importance analysis showed very little significance (correlation, mutual information, ANOVA-f) with popularity. In order to have time for the deployment part of the project, I simplified the task and developed what you can see in this folder. This actually made me more curious to investigate the idea of "song popularity" and if there is a recipe for creating a popular song. But for another project.
