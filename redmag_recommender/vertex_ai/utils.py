from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.oauth2 import service_account
from core.config import settings
import json
import numpy as np


def predict(
    instances: Union[Dict, List[Dict]],
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.

    Las instances deben ir en formato:
            instances = [
                    {"inputs": "Este es un ejemplo de texto para generar embeddings."},Ya
            ]
    """
    credentials = service_account.Credentials.from_service_account_file(
        settings.FILE_SECRET_KEY_GCP
    )

    project = settings.VERTEX_PROJECT
    endpoint_id = settings.VERTEX_ENDPOINT_ID
    location = settings.VERTEX_LOCATION

    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.

    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options,
        credentials=credentials)

    # The format of each instance should conform to the deployed model's prediction input schema.
    """instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]"""

    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        pass

    return prediction


def get_document_embedding(text: str,
                           batch_size=75):
    """
        Debido a que el LLM
        de la api solo soporta cadenas de max 75 tokens, se debe
        segmentar la petición para obtener el embeddding completo del docuemnto

    """
    tokens = text.split(' ')
    print(len(tokens), batch_size)
    if len(tokens) == 1:
        return None
    if len(tokens) < batch_size:
        try:
            instances = [
                {'inputs': text}
            ]
            return predict(instances=instances)
        except Exception as e:
            print(f'Error generando el embedding: \n\t{str(e)}')
            if '400 {"error":"Input validation error: `inputs` must have less than 75 tokens.' in str(e):
                return get_document_embedding(text=text,
                                              batch_size=batch_size-3)

            return None

    embedding_list = []
    for i in range(0, len(tokens), batch_size):
        print(i)
        try:
            if i + batch_size > len(tokens):
                inputs = " ".join(tokens[i:len(tokens)])
                print(tokens[i:len(tokens)])
                print(inputs)
            else:
                print(tokens[i:i+batch_size])
                inputs = " ".join(tokens[i:i+batch_size])
                print(inputs)
            instances = [
                {'inputs': inputs}
            ]
            embedding_list.append(
                predict(instances=instances)
            )
        except Exception as e:
            print(f'Error generando el embedding: \n\t{str(e)}')
            print(f'texto:\n\t{inputs}')
            print(f'longitud: {len(inputs.split(" "))}')
            if '400 {"error":"Input validation error: `inputs` must have less than 75 tokens.' in str(e):
                embedding_list.append(get_document_embedding(text=inputs,
                                                             batch_size=batch_size-3))
    print(embedding_list)
    return np.mean(embedding_list)
