# app.py
import streamlit as st
import pandas as pd
import backend
import chromadb
from sentence_transformers import SentenceTransformer

# -------------------------
# Page setup + Styles
# -------------------------
st.set_page_config(page_title="Chunking Optimizer", layout="wide")

st.markdown("""
    <style>
        /* Headings */
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #FF8000 !important;
        }
        .grey-text { color: #808080 !important; font-size: 14px; }

        /* Card style for main sections */
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }

        /* Sidebar progress container */
        .sidebar-card {
            background: #f7f7f7;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #ddd;
        }
        .sidebar-divider {
            border-bottom: 1px solid #ddd;
            margin: 6px 0;
        }

        /* Apple-style Soft Buttons */
        .stButton button {
            background: linear-gradient(145deg, #FF8000, #FFA64D);
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            padding: 10px 24px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.15),
                        -2px -2px 5px rgba(255,255,255,0.4);
            transition: all 0.2s ease-in-out;
        }
        .stButton button:hover {
            background: linear-gradient(145deg, #FFA64D, #FF8000);
            transform: translateY(-2px);
            box-shadow: 3px 3px 6px rgba(0,0,0,0.2),
                        -3px -3px 6px rgba(255,255,255,0.4);
        }
        .stButton button:active {
            transform: translateY(2px);
            box-shadow: inset 2px 2px 5px rgba(0,0,0,0.2),
                        inset -2px -2px 5px rgba(255,255,255,0.4);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📦 Chunking Optimizer — Sequential Flow")

# -------------------------
# Stage Progress
# -------------------------
STAGES = ["upload", "layer1", "layer2", "dtype", "quality", "chunk", "embed", "store", "retrieve"]
LABELS = {
    "upload": "📂 Upload",
    "layer1": "🧹 Preprocess L1",
    "layer2": "🧹 Preprocess L2",
    "dtype": "🔢 DTypes",
    "quality": "✅ Quality",
    "chunk": "✂️ Chunking",
    "embed": "🧩 Embedding",
    "store": "💾 Store",
    "retrieve": "🔎 Retrieval",
}

def render_progress(stage):
    st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.sidebar.title("Progress")
    for s in STAGES:
        if STAGES.index(s) < STAGES.index(stage):
            st.sidebar.markdown(f"✅ <span style='color:green'>{LABELS[s]}</span>", unsafe_allow_html=True)
        elif s == stage:
            st.sidebar.markdown(f"🟠 <span style='color:orange;font-weight:bold'>{LABELS[s]}</span>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"⚪ <span style='color:grey'>{LABELS[s]}</span>", unsafe_allow_html=True)
        st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Session init
# -------------------------
if "stage" not in st.session_state: st.session_state.stage = "upload"
if "df" not in st.session_state: st.session_state.df = None
if "chunks" not in st.session_state: st.session_state.chunks = None
if "collection" not in st.session_state: st.session_state.collection = None
if "model_obj" not in st.session_state: st.session_state.model_obj = None
if "metadatas" not in st.session_state: st.session_state.metadatas = None

def goto(s): st.session_state.stage = s
render_progress(st.session_state.stage)

# -------------------------
# Upload
# -------------------------
if st.session_state.stage == "upload":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Step 1 — Upload CSV")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            with st.spinner("Loading CSV... Please wait ⏳"):
                df = backend.load_csv(uploaded)
                st.session_state.df = df
            st.toast("✅ CSV Loaded Successfully")
            st.subheader("Preview (first 5 rows)")
            st.dataframe(backend.preview_data(df, 5))
            if st.button("Proceed to Preprocessing Layer 1"):
                goto("layer1")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Layer 1
# -------------------------
elif st.session_state.stage == "layer1":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 2 — Preprocessing Layer 1")
    if st.button("Apply Layer 1"):
        with st.spinner("Applying Layer 1..."):
            df2 = backend.layer1_preprocessing(st.session_state.df)
            st.session_state.df = df2
        st.toast("✅ Layer 1 Applied")
        st.dataframe(backend.preview_data(df2, 5))
    if st.button("Proceed to Layer 2"):
        goto("layer2")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Layer 2
# -------------------------
elif st.session_state.stage == "layer2":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 3 — Preprocessing Layer 2")
    df = st.session_state.df
    missing_total = int(df.isnull().sum().sum())
    if missing_total > 0:
        fill_number = st.number_input("Fill missing values with:", value=0.0)
    else:
        fill_number = None
    dup_count = int(df.duplicated().sum())
    dup_action = st.radio("Duplicate handling:", ["keep", "drop"]) if dup_count > 0 else "keep"
    apply_stemming = st.checkbox("Apply Stemming")
    apply_lemmatization = st.checkbox("Apply Lemmatization")
    remove_stopwords = st.checkbox("Remove Stopwords")

    if st.button("Apply Layer 2"):
        with st.spinner("Applying Layer 2..."):
            df2 = df.copy()
            if fill_number is not None: df2 = backend.handle_missing_values(df2, fill_number)
            df2 = backend.handle_duplicates(df2, dup_action)
            df2 = backend.normalize_text(df2, apply_stemming, apply_lemmatization, remove_stopwords)
            st.session_state.df = df2
        st.toast("✅ Layer 2 Applied")
        st.dataframe(backend.preview_data(df2, 5))
        st.download_button("⬇️ Download Processed CSV", df2.to_csv(index=False).encode("utf-8"), "processed.csv")
        st.subheader("Metadata Report")
        st.json(backend.generate_metadata_report(df2))
    if st.button("Proceed to DType Adjustment"):
        goto("dtype")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# DType Step
# -------------------------
elif st.session_state.stage == "dtype":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 4 — Change DTypes (Optional)")
    df = st.session_state.df
    st.table(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}))
    cols_to_change = st.multiselect("Select columns to change dtype", options=df.columns)
    dtype_map = {c: st.selectbox(f"Target dtype for {c}", ["Keep","str","int","float","datetime"], key=f"dtype_{c}") for c in cols_to_change}
    if st.button("Apply dtype changes"):
        with st.spinner("Changing dtypes..."):
            df2 = df.copy()
            applied = []
            for c,t in dtype_map.items():
                if t == "Keep": continue
                df2, err = backend.change_dtype(df2, c, t)
                if err: st.warning(f"{c} -> {t} failed: {err}")
                else: applied.append(c)
            st.session_state.df = df2
        st.toast("✅ DType Changes Applied")
        st.success(f"DType changes applied: {applied}" if applied else "No changes")
    if st.button("Proceed to Quality Gate"):
        goto("quality")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Quality Gate
# -------------------------
elif st.session_state.stage == "quality":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 5 — Quality Gate")
    df = st.session_state.df
    st.metric("Rows", len(df))
    st.metric("Columns", len(df.columns))
    st.metric("Missing values", int(df.isnull().sum().sum()))
    st.metric("Duplicate rows", int(df.duplicated().sum()))
    if st.button("Proceed to Chunking"):
        goto("chunk")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Chunking
# -------------------------
elif st.session_state.stage == "chunk":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 6 — Chunking")
    df = st.session_state.df
    choice = st.selectbox("Chunking Strategy", [
        "Fixed Size", "Recursive (LangChain)", "Semantic+Recursive", "Semantic (Cosine Similarity)"
    ])
    chunk_size = st.number_input("Chunk size", 50, 2000, 400)
    overlap = st.number_input("Overlap", 0, 500, 50)
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7) if choice == "Semantic (Cosine Similarity)" else None
    if st.button("Run Chunking"):
        with st.spinner("Creating chunks..."):
            if choice == "Fixed Size":
                chunks = backend.fixed_size_chunking_from_df(df, chunk_size, overlap)
                metadatas = None
            elif choice == "Recursive (LangChain)":
                chunks = backend.recursive_chunk(df, chunk_size, overlap)
                metadatas = backend.build_row_metadatas(df)
            elif choice == "Semantic+Recursive":
                chunks = backend.semantic_recursive_chunk(df, chunk_size, overlap)
                metadatas = backend.build_row_metadatas(df)
            else:
                chunks = backend.semantic_chunking(df, threshold=threshold)
                metadatas = backend.build_row_metadatas(df)
            st.session_state.chunks, st.session_state.metadatas = chunks, metadatas
        st.toast("✅ Chunking Completed")
        st.success(f"Created {len(chunks)} chunks")
        for c in chunks[:3]: st.code(c[:200])
    if st.session_state.chunks and st.button("Proceed to Embedding"):
        goto("embed")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Embedding
# -------------------------
elif st.session_state.stage == "embed":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 7 — Embedding")
    if st.session_state.chunks:
        model_choice = st.selectbox("Embedding model", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"])
        if st.button("Generate Embeddings"):
            with st.spinner("Generating embeddings..."):
                collection, model = backend.embed_and_store(
                    st.session_state.chunks,
                    model_name=model_choice,
                    metadatas=st.session_state.metadatas
                )
                st.session_state.collection = collection
                st.session_state.model_obj = model
            st.toast("✅ Embeddings Created")
            st.success("Embeddings created ✅")
    if st.session_state.collection and st.button("Proceed to Storage"):
        goto("store")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Store
# -------------------------
elif st.session_state.stage == "store":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 8 — Store in ChromaDB (Optional)")
    choice = st.radio("Store embeddings in ChromaDB?", ["No", "Yes"])
    if choice == "Yes":
        st.success("Stored in ChromaDB ✅")
    if st.button("Proceed to Retrieval"):
        goto("retrieve")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Retrieval
# -------------------------
elif st.session_state.stage == "retrieve":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Step 9 — Retrieval")
    q = st.text_input("Enter query")
    k = st.slider("Top-k", 1, 20, 5)
    if st.button("Search"):
        with st.spinner("Searching..."):
            res = backend.search_query(st.session_state.collection, st.session_state.model_obj, q, k)
            docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            with st.expander(f"Rank {i+1} (dist {dist:.4f})"):
                st.write(doc)
                if meta: st.json(meta)
                else: st.caption("No metadata")
    st.markdown('</div>', unsafe_allow_html=True)