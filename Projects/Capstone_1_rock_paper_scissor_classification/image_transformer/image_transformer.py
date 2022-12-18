import logging
import kserve
from PIL import Image
import requests
from io import BytesIO
import argparse
from typing import Dict
import numpy as np

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

def image_transform(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).resize((150,150)).convert('L')
    X = np.expand_dims(np.array(img, dtype=np.float32),axis=-1) / 255
    return X.tolist()

classes = ['paper', 'rock', 'scissors']

class ImageTransformer(kserve.Model):
    """ A class object for the data handling activities of Image Classification
    Task and returns a KServe compatible response.
    Args:
        kserve (class object): The Model class from the KServe
        module is passed here.
    """
    def __init__(self, name: str, predictor_host: str):
        """Initialize the model name, predictor host and the explainer host
        Args:
            name (str): Name of the model.
            predictor_host (str): The host in which the predictor runs.
            log_latency (bool): Whether to log the latency metrics per request.
        """
        super().__init__(name)
        self.predictor_host = predictor_host
        logging.info("MODEL NAME %s", name)
        logging.info("PREDICTOR URL %s", self.predictor_host)
        self.timeout = 100

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        """Pre-process activity of the Image Input data.
        Args:
            inputs (Dict): KServe http request
            headers (Dict): Kserve http request headers
        Returns:
            Dict: Returns the request input after converting it into a tensor
        """
        return {'instances': [image_transform(instance) for instance in inputs['instances']]}

    def postprocess(self, response: Dict) -> Dict:
        """Post process function of Torchserve on the KServe side is
        written here.
        Args:
            inputs (Dict): The inputs
        Returns:
            Dict: If a post process functionality is specified, it converts that into
            a new dict.
        """
        result = []
        for prediction in response['predictions']:
            output = dict(zip(classes, prediction))
            result.append(output)
        return {'predictions' : result}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )
    parser.add_argument("--model_name", help="The name of the model", required=True)

    args, _ = parser.parse_known_args()
    
    model_name = args.model_name
    host = args.predictor_host
    transformer = ImageTransformer(model_name, predictor_host=args.predictor_host)

    server = kserve.ModelServer()
    
    server.start(models=[transformer])

