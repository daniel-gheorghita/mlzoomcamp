import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np
from pydantic import BaseModel, confloat

class SongClassifierValidator(BaseModel):
    danceability: confloat(ge=0, le=1)
    energy: confloat(ge=0, le=1)

# create a runner from the saved Booster
runner = bentoml.xgboost.get("rock_alt_metal_song_genre_model:latest").to_runner()

# create a BentoML service
svc = bentoml.Service("song_classifier", runners=[runner])

# define a new endpoint on the BentoML service
@svc.api(input=JSON(pydantic_model=SongClassifierValidator), output=JSON())
async def classify_song(input):
    # use 'runner.predict.run(input)' instead of 'booster.predict'
    
    predict = await runner.predict.async_run([[input.danceability, input.energy]])
    if predict > 0.5:
        result = "rock/alternative/metal"
    else:
        result = "blues/indie/hiphop"
    return {"res":result, 'proba': predict}