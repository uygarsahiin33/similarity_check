# -*- coding: utf-8 -*-
import os, io
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, normalize
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

# ================================
# 1) Şema ve yardımcılar
# ================================
REQUIRED_COLUMNS = [
    "store_code","store_name",
    "product_id","product_name","store_format",
    "first_category","second_category","third_category","fourth_category","brand",
    "campaign_no",
    "sales_1","sales_2","sales_3","sales_4","sales_5","sales_6","sales_7",
    "sales_8","sales_9","sales_10","sales_11","sales_12","sales_13","sales_14",
    # "total sales" opsiyonel; kullanmıyoruz ama varsa kalsın
]

EPS = 1e-9

def _to_numeric(s, default=None):
    ser = s.astype(str).str.replace(",", ".", regex=False)
    out = pd.to_numeric(ser, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out

def _clean_text_cols(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": ""})
    return df

# ================================
# 2) Veri yükleme
# ================================
def load_campaign_excel(path_or_buf):
    df = pd.read_excel(path_or_buf, dtype=str)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}")

    # Satış kolonları sabit isimlerle
    day_cols = [f"sales_{i}" for i in range(1, 15)]
    for c in day_cols:
        df[c] = _to_numeric(df[c], default=0.0)

    # Metin kolonlarını toparla (yalnızca şemadakiler)
    text_cols = [
        "store_code","store_name",
        "product_id","product_name","store_format",
        "first_category","second_category","third_category","fourth_category","brand",
        "campaign_no"
    ]
    df = _clean_text_cols(df, text_cols)
    return df, day_cols

# ================================
# 3) Ürün düzeyi öznitelikler
# ================================
def fe_time_series_per_product(df, day_cols):
    id_cols = ["product_id","product_name","brand",
               "first_category","second_category","third_category","fourth_category"]

    def row_feats(s):
        arr = s[day_cols].values.astype(float)
        total = arr.sum()
        mean = arr.mean()
        std = arr.std()
        peak = arr.max()
        peak_day = int(np.argmax(arr) + 1)

        x = np.arange(1, len(arr)+1)
        x_mean = x.mean()
        y_mean = arr.mean()
        slope = np.sum((x - x_mean) * (arr - y_mean)) / (np.sum((x - x_mean)**2) + EPS)

        early = arr[:4].sum() / (total + EPS)
        mid   = arr[4:10].sum() / (total + EPS)
        late  = arr[10:].sum() / (total + EPS)

        shape = arr / (total + EPS)

        return pd.Series({
            "ts_total": total,
            "ts_mean": mean,
            "ts_std": std,
            "ts_peak": peak,
            "ts_peak_day": peak_day,
            "ts_slope": slope,
            "ts_early_share": early,
            "ts_mid_share": mid,
            "ts_late_share": late,
            **{f"ts_shape_d{d+1}": shape[d] for d in range(len(arr))}
        })

    feats = df.apply(row_feats, axis=1)
    base = pd.concat([df[id_cols], feats, df[["campaign_no","store_format"]]], axis=1)

    aggs = {c: "mean" for c in feats.columns if c != "ts_total"}
    aggs["ts_total"] = "median"
    aggs_extra = {
        "store_format": pd.Series.nunique,
        "campaign_no": pd.Series.nunique
    }

    prod = base.groupby(id_cols, as_index=False).agg({**aggs, **aggs_extra})
    prod = prod.rename(columns={
        "store_format": "n_formats",
        "campaign_no": "n_campaigns"
    })

    # Semantic text
    prod["semantic_text"] = (
        prod["brand"].fillna("") + " | " + prod["product_name"].fillna("") + " | " +
        (prod["first_category"].fillna("") + ">" +
         prod["second_category"].fillna("") + ">" +
         prod["third_category"].fillna("") + ">" +
         prod["fourth_category"].fillna(""))
    ).str.replace(">>", ">", regex=False)

    prod["product_id"] = prod["product_id"].astype(str)
    return prod

# ================================
# 4) Kampanya beraber-görülme setleri
# ================================
def build_campaign_sets(df):
    grp = df.groupby("product_id")["campaign_no"].apply(lambda s: set(s.dropna().astype(str)))
    return {str(k): v for k, v in grp.items()}

def jaccard(a, b):
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / (union + EPS)

# ================================
# 5) Model
# ================================
ALPHA, BETA, GAMMA, DELTA = 1.0, 0.6, 0.3, 1.2

CAT_COLS = ["brand","first_category","second_category","third_category","fourth_category"]
NUM_BASE_COLS = ["ts_total","ts_mean","ts_std","ts_peak","ts_peak_day","ts_slope",
                 "ts_early_share","ts_mid_share","ts_late_share",
                 "n_formats","n_campaigns"]

def _split_ts_cols(df):
    return [c for c in df.columns if c.startswith("ts_shape_d")]

class ProductSimilarityModel:
    def __init__(self, alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                 tfidf_min_df=3, tfidf_ngram=(1,2),
                 use_jaccard=True, jaccard_weight=0.2):
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.use_jaccard = use_jaccard
        self.jaccard_weight = jaccard_weight

        self.tfidf = TfidfVectorizer(min_df=tfidf_min_df, ngram_range=tfidf_ngram)
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.scaler = RobustScaler()

        self.df = None
        self.X_all = None
        self.nn = None
        self.camp_sets = None
        self.ts_cols = None

    def fit(self, df_prod: pd.DataFrame, campaign_sets=None):
        self.df = df_prod.reset_index(drop=True).copy()
        self.ts_cols = _split_ts_cols(self.df)

        X_text = self.tfidf.fit_transform(self.df["semantic_text"].astype(str))
        X_text = normalize(X_text) * self.alpha

        X_cat = self.ohe.fit_transform(self.df[CAT_COLS].astype(str))
        X_cat = normalize(X_cat) * self.beta

        num = self.df[NUM_BASE_COLS].astype(float).values
        X_num = csr_matrix(self.scaler.fit_transform(num)) * self.gamma

        X_ts = csr_matrix(self.df[self.ts_cols].astype(float).values)
        X_ts = normalize(X_ts) * self.delta

        X = hstack([X_text, X_cat, X_num, X_ts]).tocsr()
        self.X_all = normalize(X, norm="l2", axis=1, copy=False)

        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.X_all)

        self.camp_sets = campaign_sets or {}
        return self

    def _idx_by_code(self, code: str) -> int:
        arr = np.where(self.df["product_id"].astype(str).values == str(code))[0]
        if len(arr) == 0:
            raise ValueError(f"product_id bulunamadı: {code}")
        return int(arr[0])

    def search_by_code(self, product_code, k=5):
        idx = self._idx_by_code(product_code)
        q = self.X_all[idx]
        dist, ind = self.nn.kneighbors(q, n_neighbors=min(k+1, self.X_all.shape[0]))
        pairs = []
        for d, gi in zip(dist[0], ind[0]):
            if gi == idx: 
                continue
            cos_sim = 1.0 - float(d)
            if self.use_jaccard and self.camp_sets:
                a = self.camp_sets.get(str(self.df.iloc[idx]["product_id"]), set())
                b = self.camp_sets.get(str(self.df.iloc[gi]["product_id"]), set())
                jac = jaccard(a, b)
                final = (1.0 - self.jaccard_weight) * cos_sim + self.jaccard_weight * jac
            else:
                final = cos_sim
            pairs.append((gi, cos_sim, final))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:k]

    def search_all_wide(self, k=3):
        rows = []
        for i in range(len(self.df)):
            pid = self.df.iloc[i]["product_id"]
            pname = self.df.iloc[i]["product_name"]
            res = self.search_by_code(pid, k=k)
            row = {"product_id": pid, "product_name": pname}
            for rank, (gi, _, final) in enumerate(res, start=1):
                row[f"Similar {rank} ID"] = self.df.iloc[gi]["product_id"]
                row[f"Similar {rank} Name"] = self.df.iloc[gi]["product_name"]
                row[f"Similar {rank} Score (%)"] = round(final*100, 2)
            rows.append(row)
        return pd.DataFrame(rows)

# ================================
# 6) Streamlit UI
# ================================
st.set_page_config(page_title="Kampanya Satış Şekline Göre Benzer Ürünler", layout="wide")
st.title("📈 14 Gün Kampanya Satış Şekli + İçerik Benzerliği")

uploaded = st.file_uploader("Excel yükleyin", type=["xlsx"])
if uploaded:
    raw, day_cols = load_campaign_excel(uploaded)
    st.success(f"{len(raw):,} satır yüklendi • Gün kolonları: {', '.join(day_cols[:3])} ...")

    prod = fe_time_series_per_product(raw, day_cols)
    st.write(f"Ürün sayısı: {len(prod):,}")

    camp_sets = build_campaign_sets(raw)

    with st.expander("⚙️ Ağırlıklar / Ayarlar"):
        alpha = st.slider("Text (α)", 0.0, 2.0, ALPHA, 0.1)
        beta  = st.slider("Kategori (β)", 0.0, 2.0, BETA, 0.1)
        gamma = st.slider("Sayısal (γ)", 0.0, 2.0, GAMMA, 0.1)
        delta = st.slider("TS Şekil (δ)", 0.0, 2.0, DELTA, 0.1)
        use_j = st.checkbox("Kampanya Jaccard re‑rank", value=True)
        jw = st.slider("Jaccard ağırlığı (w)", 0.0, 0.8, 0.2, 0.05)

    model = ProductSimilarityModel(alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                   use_jaccard=use_j, jaccard_weight=jw).fit(prod, campaign_sets=camp_sets)

    tab1, tab2, tab3 = st.tabs(["🔍 Tek ürün arama", "📋 Toplu benzerlik tablosu", "🗺 Harita (PCA)"])

    with tab1:
        options = [f"{row['product_id']} - {row['product_name']}" for _, row in prod.iterrows()]
        sel = st.selectbox("Ürün seç", options)
        sel_code = sel.split(" - ", 1)[0]
        try:
            k = st.slider("Kaç benzer gösterilsin?", 1, 15, 10)
            res = model.search_by_code(sel_code, k=k)
            base = prod[prod["product_id"] == sel_code].iloc[0]
            st.subheader("Aranan Ürün")
            st.dataframe(base.to_frame().T)

            rows = []
            for gi, cos_sim, final in res:
                r = prod.iloc[gi][["product_id","product_name","brand",
                                   "first_category","second_category","third_category","fourth_category"]].copy()
                r["similarity_cosine"] = round(cos_sim, 4)
                r["similarity_final(%)"] = round(final*100, 2)
                rows.append(r)
            st.subheader("Benzer Ürünler")
            st.dataframe(pd.DataFrame(rows))
        except Exception as e:
            st.error(str(e))

    with tab2:
        if st.button("Tüm ürünlerin benzerlerini hesapla"):
            all_df = model.search_all_wide(k=3)
            st.write(f"Toplam {len(all_df)} ürün")
            st.dataframe(all_df)
            buf = io.BytesIO()
            all_df.to_excel(buf, index=False)
            buf.seek(0)
            st.download_button("📥 Excel indir", data=buf,
                               file_name="similar_products_all.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab3:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(model.X_all.toarray())
        pdf = pd.DataFrame(coords, columns=["x","y"])
        pdf["product_id"] = prod["product_id"]
        pdf["product_name"] = prod["product_name"]

        sel2 = st.selectbox("Haritada öne çıkar", [f"{c} - {n}" for c,n in zip(prod["product_id"], prod["product_name"])], key="map_sel")
        sel_code2 = sel2.split(" - ", 1)[0]
        sel_idx = np.where(prod["product_id"].astype(str).values == str(sel_code2))[0][0]
        nn = model.search_by_code(sel_code2, k=3)
        nn_idx = [gi for gi,_,_ in nn]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pdf["x"], y=pdf["y"], mode="markers",
                                 marker=dict(size=6, color="rgba(200,200,200,0.5)"),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=pdf.iloc[nn_idx]["x"], y=pdf.iloc[nn_idx]["y"],
                                 mode="markers", marker=dict(size=10, color="#e96411"),
                                 hoverinfo="skip", showlegend=False))
        sx, sy = pdf.iloc[sel_idx]["x"], pdf.iloc[sel_idx]["y"]
        fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers",
                                 marker=dict(size=12, color="#0D0DD4"),
                                 hoverinfo="skip", showlegend=False))

        base = prod.iloc[sel_idx]
        sel_info = (f"<b>{base['product_id']}</b> — {base['product_name']}<br>"
                    f"{base['brand']}<br>"
                    f"{base['first_category']} > {base['second_category']} > "
                    f"{base['third_category']} > {base['fourth_category']}")
        fig.add_annotation(x=sx, y=sy, xanchor="left", yanchor="bottom",
                           xshift=12, yshift=12, showarrow=False,
                           text=sel_info, align="left",
                           font=dict(size=12, color="#fff"),
                           bgcolor="rgba(0,0,0,0.85)", bordercolor="#0D0DD4",
                           borderwidth=1, borderpad=6)

        x_panel = pdf["x"].max() + (pdf["x"].max()-pdf["x"].min())*0.2
        y_top   = pdf["y"].max()
        y_gap   = 0.5*(pdf["y"].max()-pdf["y"].min())
        for i, gi in enumerate(nn_idx, start=1):
            r = prod.iloc[gi]
            info = (f"<b>{r['product_id']}</b> — {r['product_name']}<br>{r['brand']}<br>"
                    f"{r['first_category']} > {r['second_category']} > "
                    f"{r['third_category']} > {r['fourth_category']}")
            fig.add_annotation(x=x_panel, y=y_top - (i-1)*y_gap,
                               xanchor="left", yanchor="top",
                               text=info, showarrow=False, align="left",
                               font=dict(size=12, color="#fff"),
                               bgcolor="rgba(0,0,0,0.85)", bordercolor="#e96411",
                               borderwidth=1, borderpad=6)

        fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), hovermode=False)
        st.plotly_chart(fig, use_container_width=True)
