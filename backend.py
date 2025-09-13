# backend.py
import io, csv, typing, re, datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# CSV Loading / Utilities
# ----------------------------
try:
    import chardet
except Exception:
    chardet = None

def _detect_sep(sample_text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096])
        return dialect.delimiter
    except Exception:
        return ","

def load_csv(file_or_path: typing.Union[str, io.BytesIO, io.StringIO]):
    if isinstance(file_or_path, str):
        return pd.read_csv(file_or_path)

    if hasattr(file_or_path, "read"):
        file_or_path.seek(0)
        raw = file_or_path.read()

        if isinstance(raw, (bytes, bytearray)):
            encoding = "utf-8"
            if chardet:
                try:
                    res = chardet.detect(raw)
                    encoding = res.get("encoding") or "utf-8"
                except Exception:
                    pass
            text = raw.decode(encoding, errors="replace")
            sep = _detect_sep(text)
            return pd.read_csv(io.StringIO(text), sep=sep)

        if isinstance(raw, str):
            sep = _detect_sep(raw)
            return pd.read_csv(io.StringIO(raw), sep=sep)

    raise ValueError("Unsupported input for load_csv")

def preview_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(min(len(df), n)).reset_index(drop=True)

def change_dtype(df: pd.DataFrame, col: str, dtype: str):
    try:
        if dtype == "str":
            df[col] = df[col].astype(str)
        elif dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="raise").astype("Int64")
        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="raise")
        else:
            return df, f"Unsupported dtype: {dtype}"
        return df, None
    except Exception as e:
        return df, str(e)

# ----------------------------
# Preprocessing
# ----------------------------
def layer1_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    # Fix empty column names
    df2.columns = [c if c.strip() != "" else "empty_column" for c in df2.columns]
    # Remove HTML
    clean_re = re.compile(r"<.*?>")
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).apply(lambda s: re.sub(clean_re, "", s))
    # Lowercase + strip whitespace
    for c in df2.select_dtypes(include=["object"]).columns:
        df2[c] = df2[c].astype(str).str.lower().str.strip()
    return df2

def handle_missing_values(df: pd.DataFrame, fill_number: float):
    return df.fillna(fill_number)

def handle_duplicates(df: pd.DataFrame, action: str):
    if action == "drop":
        return df.drop_duplicates().reset_index(drop=True)
    return df

def normalize_text(df: pd.DataFrame, apply_stemming=False, apply_lemmatization=False, remove_stopwords=False):
    df2 = df.copy()
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words("english"))

    for c in df2.select_dtypes(include=["object"]).columns:
        processed = []
        for text in df2[c].astype(str):
            words = text.split()
            if remove_stopwords:
                words = [w for w in words if w not in stops]
            if apply_stemming:
                words = [ps.stem(w) for w in words]
            if apply_lemmatization:
                words = [lemmatizer.lemmatize(w) for w in words]
            processed.append(" ".join(words))
        df2[c] = processed
    return df2

def generate_metadata_report(df: pd.DataFrame) -> dict:
    report = {}
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    report["text_info"] = {
        "count": len(text_cols),
        "columns": text_cols,
        "rows": len(df)
    }
    num_info = {}
    for c in num_cols:
        num_info[c] = {
            "min": df[c].min(),
            "max": df[c].max(),
            "mean": df[c].mean(),
            "nulls": df[c].isnull().sum()
        }
    report["numeric_info"] = num_info
    report["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return report

# ----------------------------
# Chunking
# ----------------------------
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

def fixed_size_chunking_from_df(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    docs = df.astype(str).apply(lambda row: " | ".join([f"{c}:{row[c]}" for c in df.columns]), axis=1).tolist()
    text = "\n".join(docs)
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i:i+chunk_size])
    return chunks

def recursive_chunk(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    docs = df.astype(str).apply(lambda row: ", ".join([f"{c}:{row[c]}" for c in df.columns]), axis=1).tolist()
    text = "\n".join(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def semantic_compression(df: pd.DataFrame):
    compressed_rows = []
    for _, row in df.iterrows():
        pieces = [f"{c}:{str(row[c])[:30]}" for c in df.columns]
        compressed_rows.append("; ".join(pieces))
    return compressed_rows


def semantic_recursive_chunk(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50):
    compressed = semantic_compression(df)
    text = "\n".join(compressed)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def semantic_chunking(df: pd.DataFrame, model_name="all-MiniLM-L6-v2", threshold=0.7):
    docs = df.astype(str).apply(lambda row: " | ".join([f"{c}:{row[c]}" for c in df.columns]), axis=1).tolist()
    if not docs:
        return []
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=False)

    chunks, current_chunk, current_embs = [], [], []
    for text, emb in zip(docs, embeddings):
        if not current_chunk:
            current_chunk.append(text)
            current_embs.append(emb)
        else:
            avg_emb = np.mean(current_embs, axis=0).reshape(1, -1)
            sim = cosine_similarity(avg_emb, emb.reshape(1, -1))[0][0]
            if sim >= threshold:
                current_chunk.append(text)
                current_embs.append(emb)
            else:
                chunks.append("\n".join(current_chunk))
                current_chunk, current_embs = [text], [emb]
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

# ----------------------------
# Embedding + Storage
# ----------------------------
def embed_and_store(chunks, model_name="all-MiniLM-L6-v2", chroma_path="chromadb_store", collection_name="default", metadatas=None):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)
    emb_lists = [list(map(float, e)) for e in embeddings]
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(collection_name)
    try:
        existing = collection.get()
        if "ids" in existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except:
        pass
    ids = [str(i) for i in range(len(chunks))]
    if metadatas and len(metadatas) == len(chunks):
        collection.add(ids=ids, documents=chunks, embeddings=emb_lists, metadatas=metadatas)
    else:
        collection.add(ids=ids, documents=chunks, embeddings=emb_lists)
    return collection, model

def search_query(collection, model, query: str, k: int = 5):
    q_emb = model.encode([query])
    return collection.query(query_embeddings=q_emb, n_results=k, include=["documents", "metadatas", "distances"])

def build_row_metadatas(df: pd.DataFrame):
    metadatas = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for _, row in df.iterrows():
        md = {}
        for c in numeric_cols:
            try:
                md[c] = float(row[c]) if not pd.isna(row[c]) else None
            except:
                md[c] = None
        metadatas.append(md)
    return metadatas