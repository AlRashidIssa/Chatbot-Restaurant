import sys
import pytest
from fastapi.testclient import TestClient
import time
sys.path.append("/workspaces/Chatbot-Restaurant/")

# Assuming app is your FastAPI instance
from src.api.app import app  
from src.utils import chatbot_log  # Assuming this is your logging utility
from src.utils.ingest_query_database import IngestQueryDatabase
from src.preprocess.combined_tables import CombinedTables
from src.preprocess.apply_embedding_combined import EmbeddingForCombined
from src.preprocess.faiss_index import FAISSIndex
from src.models.embedding_model_all_miniLM_L6_v2 import EmbeddingLoader
from src.models.huggneface_model_flanT5 import FlanT5Load
from src.rag.retrieval import Retrieve
from src.rag.respons_generation import GenerateResponse
from src.config import Config


# Initialize TestClient
client = TestClient(app)
CONFIGRATION = Config(None)


@pytest.fixture(scope="module")
def ingest_data():
    """Fixture to set up and return processed data."""
    INGESTDATA = IngestQueryDatabase()
    FAQsDF = INGESTDATA.ingest(db_path=CONFIGRATION.path_database, query=CONFIGRATION.FAQsQ)
    MENUITEMsDF = INGESTDATA.ingest(db_path=CONFIGRATION.path_database, query=CONFIGRATION.MENUITEMsQ)
    
    FAQsDF = FAQsDF.drop_duplicates()
    MENUITEMsDF = MENUITEMsDF.drop_duplicates()
    
    COMBINEDCOLUMNS = CombinedTables()
    FAQsDF_c = COMBINEDCOLUMNS.combined(columns=CONFIGRATION.columns_faqs, df=FAQsDF)
    MENUITEMsDF_c = COMBINEDCOLUMNS.combined(columns=CONFIGRATION.columns_menuitems, df=MENUITEMsDF)

    EMBEDDINGMODEL = EmbeddingLoader().load()
    EMBEDDINGCOBINED = EmbeddingForCombined()

    FAQs_E = EMBEDDINGCOBINED.embedded(embedding_model=EMBEDDINGMODEL, df=FAQsDF_c)
    MENUITEMs_E = EMBEDDINGCOBINED.embedded(embedding_model=EMBEDDINGMODEL, df=MENUITEMsDF_c)

    FAISSINDEX = FAISSIndex()
    FAQsIndex = FAISSINDEX.create_faiss_index(embedding_array=FAQs_E)
    MENUITEMsIndex = FAISSINDEX.create_faiss_index(embedding_array=MENUITEMs_E)
    
    return {
        'FAQsIndex': FAQsIndex,
        'MENUITEMsIndex': MENUITEMsIndex,
        'FAQsDF_c': FAQsDF_c,
        'MENUITEMsDF_c': MENUITEMsDF_c,
        'embedding_model': EMBEDDINGMODEL,
        'flan_model': FlanT5Load().load()
    }

def test_ingest_data(ingest_data):
    """Test data ingestion."""
    assert len(ingest_data['FAQsDF_c']) > 0
    assert len(ingest_data['MENUITEMsDF_c']) > 0
    assert ingest_data['FAQsIndex'] is not None
    assert ingest_data['MENUITEMsIndex'] is not None

def test_embedding_generation(ingest_data):
    """Test embedding generation."""
    
    # Check if the combined DataFrames are non-empty
    assert ingest_data['FAQsDF_c'].shape[0] > 0, "FAQsDF_c is empty"
    assert ingest_data['MENUITEMsDF_c'].shape[0] > 0, "MENUITEMsDF_c is empty"
    
    # Check if the FAISS indices are not empty (we assume they are generated as a non-empty index object)
    assert ingest_data['FAQsIndex'] is not None, "FAQsIndex is None"
    assert ingest_data['MENUITEMsIndex'] is not None, "MENUITEMsIndex is None"
    
    # Optionally, you can add more checks specific to FAISS index properties, like number of entries
    # Assuming the FAISS index has a 'ntotal' attribute for the total number of vectors
    assert ingest_data['FAQsIndex'].ntotal > 0, "FAQsIndex has no entries"
    assert ingest_data['MENUITEMsIndex'].ntotal > 0, "MENUITEMsIndex has no entries"


def test_retrieval(ingest_data):
    """Test data retrieval using the FAISS index."""
    query = "What is the menu?"
    retriever = Retrieve().retrieve(query=query,
                                     embedding_model=ingest_data['embedding_model'],
                                     index_1=ingest_data['FAQsIndex'],
                                     index_2=ingest_data['MENUITEMsIndex'],
                                     df_index_1=ingest_data['FAQsDF_c'],
                                     df_index_2=ingest_data['MENUITEMsDF_c'],
                                     top_k=5)
    
    assert len(retriever) > 0

def test_generate_response(ingest_data):
    """Test the response generation."""
    query = "What is the menu?"
    retriever = Retrieve().retrieve(query=query,
                                     embedding_model=ingest_data['embedding_model'],
                                     index_1=ingest_data['FAQsIndex'],
                                     index_2=ingest_data['MENUITEMsIndex'],
                                     df_index_1=ingest_data['FAQsDF_c'],
                                     df_index_2=ingest_data['MENUITEMsDF_c'],
                                     top_k=5)
    
    response = GenerateResponse().generate(query=query,
                                           retriever=retriever,
                                           generate_model=ingest_data['flan_model'],
                                           max_length=CONFIGRATION.max_length,
                                           do_sample=CONFIGRATION.do_sample,
                                           temperature=CONFIGRATION.temperature,
                                           top_p=CONFIGRATION.top_p,
                                           top_k=CONFIGRATION.top_k)
    assert response is not None
    assert isinstance(response, str)

def test_chatbot_response():
    """Test the /chat endpoint."""
    query = "What is the menu?"
    response = client.post("/chat", data={"query": query})
    
    assert response.status_code == 200
    assert "response" in response.text  