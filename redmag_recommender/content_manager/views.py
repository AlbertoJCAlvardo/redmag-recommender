from django.shortcuts import render
from django.http import JsonResponse, request
from django.views.decorators.csrf import csrf_exempt
from core.config import settings
from db.utils import ChromaManager
from vertex_ai.utils import get_document_embedding, predict
import json
import numpy as np
# Create your views here.


@csrf_exempt
def insert_human_med(request: request):
    """
        Vectoriza y almacena un med generado por humanos
    """
    api_key = request.headers.get('X-API-KEY')
    if api_key != settings.API_KEY:
        return JsonResponse({'error': 'Invalid API Key'},
                            status=401)

    if request.method == 'POST':
        body = json.loads(request.body)
        content = body['content']
        embedding_resp = get_document_embedding(content)

        print(f'embedding_response: {embedding_resp}')

        cm = ChromaManager()
        cm.add_document(collection_name='meds_humanos',
                        document=content,
                        metadata={
                            'foreign_id': body['id']
                        })

        return JsonResponse({'message': 'Planeacion agregada con exito'},
                            status=200)


@csrf_exempt
def insert_vertex_med(request: request):
    """
        Vectoriza y almacena un med generado por humanos
    """
    api_key = request.headers.get('X-API-KEY')
    if api_key != settings.API_KEY:
        return JsonResponse({'error': 'Invalid API Key'},
                            status=401)

    if request.method == 'POST':
        body = json.loads(request.body)
        text_dic = body['text']
        candidate = text_dic['candidates'][0]
        content_dic = candidate['content']['parts'][0]['text']
        text = json.loads(content_dic)
        content = f'{text['título']}\n{text['descripción']}'
        print(content)
        print(content.split(' '))
        embedding_resp = get_document_embedding(content)

        print(f'embedding_response: {embedding_resp}')

        cm = ChromaManager()
        cm.add_document(collection_name='meds_vertex',
                        document=content,
                        metadata={
                            'foreign_id': body['id']
                        })

        return JsonResponse({'message': 'Planeacion agregada con exito'},
                            status=200)


@csrf_exempt
def insert_planea(request: request):
    """
        Vectoriza y almacena un med generado por humanos
    """
    api_key = request.headers.get('X-API-KEY')
    if api_key != settings.API_KEY:
        return JsonResponse({'error': 'Invalid API Key'},
                            status=401)

    if request.method == 'POST':
        body = json.loads(request.body)
        content = body['content']
        embedding_resp = get_document_embedding(content)
        print(f'embedding_response: {embedding_resp}')

        cm = ChromaManager()
        cm.add_document(collection_name='planeaciones',
                        document=content,
                        metadata={
                            'foreign_id': body['id']
                        })

        return JsonResponse({'message': 'Planeacion agregada con exito'},
                            status=200)


def recommend_planeaciones(request: request):
    """
        Dada una lista de planeaciones vistas recientemente por 
        el usuario, 
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Bad request'},
                            status=400)

    try:
        body = json.loads(request.body)
        recents = body['recents']
        cm = ChromaManager()
        embedding_list = [
            cm.query_by_metadata(collection_name='planeaciones_redmagia',
                                 key='foreign_id',
                                 value=x) for x in recents
        ]
        mean_vector = np.mean(embedding_list)
        similars = cm.query_vector(collection_name='pdas',
                                   embedding=mean_vector
                                   )

        print(similars)

        return JsonResponse(similars,
                            status=200)
    except Exception as e:
        print(e)
        return JsonResponse({'error': 'error xd'},
                            status=400)
