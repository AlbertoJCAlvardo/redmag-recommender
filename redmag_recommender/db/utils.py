import chromadb
import os
import json
import uuid
from core.config import settings
from vertex_ai.utils import get_document_embedding


class ChromaManager:
    def __init__(self, path="./chroma_db", debug=False):
        """
        Inicializa el cliente de ChromaDB y las colecciones.
        :param path: Ruta donde se almacenarán los datos de ChromaDB.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collections = {
            "planeaciones_redmagia": self._get_or_create_collection("planeaciones_redmagia"),
            "meds_vertex": self._get_or_create_collection("meds_vertex"),
            "meds_humanos": self._get_or_create_collection("meds_humanos"),
            "pdas": self._get_or_create_collection("pdas")
        }

    def debug_print(self, *args):
        """
            Prints something if it is in debug mode
        """
        printable = "".join(
            [x + " " for x in args])
        print(printable)

    def _get_or_create_collection(self, name):
        """
        Obtiene una colección existente o la crea si no existe.
        :param name: Nombre de la colección.
        :return: Colección de ChromaDB.
        """
        try:
            collection = self.client.get_collection(name=name)
            print(f"Colección '{name}' obtenida correctamente.")
        except Exception as e:
            print(
                f"Colección '{name}' no encontrada. Creando nueva colección...")
            collection = self.client.create_collection(name=name)
            if name == 'pdas':
                print('Metiendole los pdas')
                self._get_pdas(collection)
        return collection

    def add_document(self, collection_name, document, metadata=None, doc_id=None):
        """
        Añade un documento a una colección.
        :param collection_name: Nombre de la colección.
        :param document: Texto del documento.
        :param metadata: Metadatos asociados al documento (opcional).
        :param doc_id: ID único del documento (opcional).
        """
        if collection_name not in self.collections:
            raise ValueError(f"Colección '{collection_name}' no existe.")

        self.collections[collection_name].add(
            documents=[document],
            metadatas=[metadata] if metadata else None,
            ids=[doc_id] if doc_id else [str(uuid.uuid4())]
        )
        print(f"Documento añadido a la colección '{collection_name}'.")

    def query_collection(self, collection_name, query_text, n_results=5):
        """
        Realiza una búsqueda en una colección.
        :param collection_name: Nombre de la colección.
        :param query_text: Texto de búsqueda.
        :param n_results: Número de resultados a devolver.
        :return: Resultados de la búsqueda.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Colección '{collection_name}' no existe.")

        results = self.collections[collection_name].query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def query_vector(self, collection_name, embedding, n_results=5):
        """
            Obtiene una lista de elementos similares a
            los provistos de acuerdo a un vector representativo.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Colección '{collection_name}' no existe.")

        results = self.collections[collection_name].query(
            query_vector=embedding,
            n_results=n_results
        )

        return results

    def query_by_metadata(self, collection_name,
                          key, value, n_results=5):
        """
        Realiza una búsqueda en una colección.
        : param collection_name: Nombre de la colección.
        : param key: nombre del campo del metadata para hacer el filtro
        : param value: valor del metadata para hacer el filtro
        : param n_results: Número de resultados a devolver.
        : return: Resultados de la búsqueda.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Colección '{collection_name}' no existe.")

        if not key or not value:
            raise ValueError(f"Faltan key o value")

        results = self.collections[collection_name].query(
            query_texts=None,
            n_results=n_results,
            where={key: value}
        )
        return results

    def delete_document(self, collection_name, doc_id):
        """
        Elimina un documento de una colección.
        : param collection_name: Nombre de la colección.
        : param doc_id: ID del documento a eliminar.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Colección '{collection_name}' no existe.")

        self.collections[collection_name].delete(ids=[doc_id])
        print(
            f"Documento con ID '{doc_id}' eliminado de la colección '{collection_name}'.")

    def _get_pdas(self, collection):
        """
            Llena la base de datos de los pda's 
            con los archivos provistos por juan carlos.
        """
        try:
            with open(settings.FILE_PDA_SEC, 'r') as f:
                pda_s = json.loads(f.read())
                f.close()

            with open(settings.FILE_PDA_NSEC, 'r') as f:
                pda_ns = json.loads(f.read())
                f.close()

            general_pda = list()
            for _ in pda_s + pda_ns:
                general_pda.append(_['learningProgressionContent'])

            pda_list = list(set(general_pda))
            pda_embed = dict(
                map(lambda x: (x, get_document_embedding(x)), pda_list))
            for element in pda_s:
                collection.add(
                    documents=[element['learningProgressionContent']],
                    metadatas=element,
                    ids=[str(uuid.uuid4())]
                )
            for element in pda_ns:
                collection.add(
                    embeddings=pda_embed[element['learningProgressionContent']],
                    documents=[element['learningProgressionContent']],
                    metadatas=element,
                    ids=[str(uuid.uuid4())]
                )
            self.debug_print('Base de datos de planeaciones creadas con exito')

        except Exception as e:
            self.debug_print(
                f'Error leyendo los archivos de pdas: \n\t{str(e)}\n\t{repr(e)}')

        return
