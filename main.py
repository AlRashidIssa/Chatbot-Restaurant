import os
import sys
import time
# Get the absolute path to the directory one level above the current file's directory
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from src.utils import chatbot_log, error_log
from src.utils.ingest_query_database import IngestQueryDatabase
from src.preprocess.combined_tables import CombinedTables
from src.preprocess.apply_embedding_combined import EmbeddingForCombined
from src.preprocess.faiss_index import FAISSIndex
from src.models.embedding_model_all_miniLM_L6_v2 import EmbeddingLoader
from src.models.huggneface_model_flanT5 import FlanT5Load
from src.rag.retrieval import Retrieve 
from src.rag.respons_generation import GenerateResponse
from src.config import Config

FLANMODEL = FlanT5Load().load()
EMBEDDINGMODEL = EmbeddingLoader().load()

INGESTDATA = IngestQueryDatabase()
COMBINEDCOLUMNS = CombinedTables()
EMBEDDINGCOBINED = EmbeddingForCombined()
FAISSINDEX = FAISSIndex()

CONFIGRATION = Config(None)

FAQsDF = INGESTDATA.ingest(db_path=CONFIGRATION.path_database, query=CONFIGRATION.FAQsQ)
MENUITEMsDF = INGESTDATA.ingest(db_path=CONFIGRATION.path_database, query=CONFIGRATION.MENUITEMsQ)

# Hanling Duplicates
FAQsDF = FAQsDF.drop_duplicates()
MENUITEMsDF = MENUITEMsDF.drop_duplicates()

# Combiend
FAQsDF_c = COMBINEDCOLUMNS.combined(columns=CONFIGRATION.columns_faqs, df=FAQsDF)
MENUITEMsDF_c = COMBINEDCOLUMNS.combined(columns=CONFIGRATION.columns_menuitems, df=MENUITEMsDF)

# Generative Embedding
FAQs_E = EMBEDDINGCOBINED.embedded(embedding_model=EMBEDDINGMODEL, df=FAQsDF_c)
MENUITEMs_E = EMBEDDINGCOBINED.embedded(embedding_model=EMBEDDINGMODEL, df=MENUITEMsDF_c)

# Convert FAISS Index for FAQs & Menu Items
FAQsIndex = FAISSINDEX.create_faiss_index(embedding_array=FAQs_E)
MENUITEMsIndex = FAISSINDEX.create_faiss_index(embedding_array=MENUITEMs_E)

def main(query: str) -> str:
    """
    
    """
    # Retrieval
    RETRIEVE = Retrieve().retrieve(query=query,
                                   embedding_model=EMBEDDINGMODEL,
                                   index_1=FAQsIndex,
                                   index_2=MENUITEMsIndex,
                                   df_index_1=FAQsDF_c,
                                   df_index_2=MENUITEMsDF_c,
                                   top_k=CONFIGRATION.top_k_results)
    # Model Generative
    RESPONSEs = GenerateResponse().generate(query=query,
                                            retriever=RETRIEVE,
                                            generate_model=FLANMODEL,
                                            max_length=CONFIGRATION.max_length,
                                            do_sample=CONFIGRATION.do_sample,
                                            temperature=CONFIGRATION.temperature,
                                            top_p=CONFIGRATION.top_p,
                                            top_k=CONFIGRATION.top_k)
    return RESPONSEs


if __name__ == "__main__":
    for _ in range(10):
        query = input("Ask me any thinks:")
        response = main(query=query)
        print("\nChatbot Response:")
        print(response)
        print(time)


