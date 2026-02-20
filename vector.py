import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# =========================
# CONFIG
# =========================
CSV_FILE = "cameras.csv"
DB_LOCATION = "./chroma_shopping_db"
COLLECTION_NAME = "camera_products"

os.makedirs(DB_LOCATION, exist_ok=True)

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()   # remove extra spaces
df = df.fillna("Not available")

print("📄 CSV Columns:", df.columns.tolist())

# =========================
# FLEXIBLE COLUMN MAPPING
# =========================
def find_col(possible_names):
    for col in df.columns:
        if col.lower() in possible_names:
            return col
    return None

COL_PRODUCT = find_col(["productname", "product_name", "name", "title", "model"])
COL_BRAND = find_col(["brand", "company", "manufacturer"])
COL_CATEGORY = find_col(["category", "type"])
COL_PRICE = find_col(["price", "cost", "amount"])
COL_RATING = find_col(["rating", "stars", "review"])
COL_DESC = find_col(["description", "details", "specs"])

# =========================
# EMBEDDINGS
# =========================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# =========================
# VECTOR STORE
# =========================
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

# =========================
# INGEST ONLY ONCE
# =========================
if vector_store._collection.count() == 0:
    print("📥 Ingesting camera data...")

    documents = []

    for _, row in df.iterrows():
        content = f"""
        Product: {row.get(COL_PRODUCT, 'N/A')}
        Brand: {row.get(COL_BRAND, 'N/A')}
        Category: {row.get(COL_CATEGORY, 'N/A')}
        Price: {row.get(COL_PRICE, 'N/A')}
        Rating: {row.get(COL_RATING, 'N/A')}
        Description: {row.get(COL_DESC, 'N/A')}
        """

        documents.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "product": row.get(COL_PRODUCT, "N/A"),
                    "brand": row.get(COL_BRAND, "N/A"),
                    "price": row.get(COL_PRICE, "N/A"),
                    "rating": row.get(COL_RATING, "N/A"),
                }
            )
        )

    vector_store.add_documents(documents)
    print("✅ Data successfully stored in ChromaDB")

else:
    print("✅ Using existing embeddings")

# =========================
# RETRIEVER
# =========================
retriever = vector_store.as_retriever(search_kwargs={"k": 5})