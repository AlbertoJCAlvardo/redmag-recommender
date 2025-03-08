from dotenv import load_dotenv
from redmag_recommender.settings import BASE_DIR
import os

load_dotenv()


class Settings():
    API_KEY = os.getenv('API_KEY')
    VERTEX_PROJECT = os.getenv('VERTEX_PROJECT')
    VERTEX_ENDPOINT_ID = os.getenv('VERTEX_ENDPOINT_ID')
    VERTEX_LOCATION = os.getenv('VERTEX_LOCATION')
    FILE_SECRET_KEY_GCP = os.path.join(BASE_DIR,
                                       os.getenv('FILE_SECRET_KEY_GCP'))

    FILE_PDA_SEC = os.path.join(BASE_DIR,
                                os.getenv('FILE_PDA_SEC'))

    FILE_PDA_NSEC = os.path.join(BASE_DIR,
                                 os.getenv('FILE_PDA_NSEC'))


settings = Settings()
