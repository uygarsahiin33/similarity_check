# -*- coding: utf-8 -*-
import os, io
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
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
# 3) Ürün düzeyi öznitelikler (VEKTÖRİZE)
# ================================
def fe_time_series_per_product(df, day_cols, store_filter: str | None = None):
    # Mağaza filtresi (opsiyonel)
    if store_filter is not None:
        df = df[df["store_code"].astype(str) == str(store_filter)].copy()

    id_cols = ["product_id","product_name","brand",
               "first_category","second_category","third_category","fourth_category"]

    # ---- Vektörize zaman serisi istatistikleri ----
    arr = df[day_cols].to_numpy(dtype=np.float32)  # float32 -> daha az RAM, daha hızlı
    n = arr.shape[1]
    x = np.arange(1, n+1, dtype=np.float32)
    x_centered = x - x.mean()

    total = arr.sum(axis=1)
    mean  = arr.mean(axis=1)
    std   = arr.std(axis=1)
    peak  = arr.max(axis=1)
    peak_day = arr.argmax(axis=1).astype(np.int32) + 1

    # slope = cov(x, y) / var(x)
    y_centered = arr - mean[:, None]
    denom = (x_centered**2).sum() + EPS
    slope = (y_centered * x_centered).sum(axis=1) / denom

    # dönem payları
    early = arr[:, :4].sum(axis=1)   / (total + EPS)
    mid   = arr[:, 4:10].sum(axis=1) / (total + EPS)
    late  = arr[:, 10:].sum(axis=1)  / (total + EPS)

    # shape (d1..d14)
    shape = (arr / (total[:, None] + EPS)).astype(np.float32)
    shape_df = pd.DataFrame(
        shape, columns=[f"ts_shape_d{i+1}" for i in range(n)]
    )

    feats = pd.DataFrame({
        "ts_total": total.astype(np.float32),
        "ts_mean":  mean.astype(np.float32),
        "ts_std":   std.astype(np.float32),
        "ts_peak":  peak.astype(np.float32),
        "ts_peak_day": peak_day.astype(np.int32),
        "ts_slope": slope.astype(np.float32),
        "ts_early_share": early.astype(np.float32),
        "ts_mid_share":   mid.astype(np.float32),
        "ts_late_share":  late.astype(np.float32),
    })

    feats = pd.concat([feats, shape_df], axis=1)

    base = pd.concat([df[id_cols].reset_index(drop=True),
                      feats.reset_index(drop=True),
                      df[["campaign_no","store_format"]].reset_index(drop=True)], axis=1)

    aggs = {c: "mean" for c in feats.columns if c != "ts_total"}
    aggs["ts_total"] = "median"
    aggs_extra = {"store_format": pd.Series.nunique, "campaign_no": pd.Series.nunique}

    prod = base.groupby(id_cols, as_index=False).agg({**aggs, **aggs_extra})
    prod = prod.rename(columns={"store_format": "n_formats","campaign_no": "n_campaigns"})

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
def build_campaign_sets(df, store_filter: str | None = None):
    if store_filter is not None:
        df = df[df["store_code"].astype(str) == str(store_filter)].copy()
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
        # OneHotEncoder: sürüm uyumluluğu
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

        self.scaler = RobustScaler()

        self.df = None
        self.X_all = None
        self.nn = None
        self.camp_sets = None
        self.ts_cols = None

        # Performans: O(1) id->index
        self._id2idx = None
        self._pid_arr = None

    def fit(self, df_prod: pd.DataFrame, campaign_sets=None):
        self.df = df_prod.reset_index(drop=True).copy()
        self.ts_cols = _split_ts_cols(self.df)

        X_text = self.tfidf.fit_transform(self.df["semantic_text"].astype(str))
        X_text = normalize(X_text).astype(np.float32) * self.alpha

        X_cat = self.ohe.fit_transform(self.df[CAT_COLS].astype(str))
        X_cat = normalize(X_cat).astype(np.float32) * self.beta

        num = self.df[NUM_BASE_COLS].astype(np.float32).to_numpy()
        X_num = csr_matrix(self.scaler.fit_transform(num).astype(np.float32)) * self.gamma

        X_ts = csr_matrix(self.df[self.ts_cols].astype(np.float32).to_numpy())
        X_ts = normalize(X_ts).astype(np.float32) * self.delta

        X = hstack([X_text, X_cat, X_num, X_ts]).tocsr()
        self.X_all = normalize(X, norm="l2", axis=1, copy=False)

        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.X_all)

        self.camp_sets = campaign_sets or {}

        # O(1) id->index
        self._pid_arr = self.df["product_id"].astype(str).to_numpy()
        self._id2idx = {pid: i for i, pid in enumerate(self._pid_arr)}
        return self

    def _idx_by_code(self, code: str) -> int:
        if self._id2idx is None:
            arr = np.where(self.df["product_id"].astype(str).values == str(code))[0]
            if len(arr) == 0:
                raise ValueError(f"product_id bulunamadı: {code}")
            return int(arr[0])
        idx = self._id2idx.get(str(code))
        if idx is None:
            raise ValueError(f"product_id bulunamadı: {code}")
        return int(idx)

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
        # Toplu komşu hesabı: n_neighbors=k+1 (kendisi + k komşu)
        n = self.X_all.shape[0]
        n_neighbors = min(k + 1, n)
        try:
            dist, ind = self.nn.kneighbors(self.X_all, n_neighbors=n_neighbors, return_distance=True)
        except TypeError:
            dist, ind = self.nn.kneighbors(self.X_all, n_neighbors=n_neighbors)

        rows = []
        for i in range(n):
            pid = self.df.iloc[i]["product_id"]
            pname = self.df.iloc[i]["product_name"]
            # ilk indeks kendisi olabilir -> filtrele
            cand = [(gi, 1.0 - float(d)) for d, gi in zip(dist[i], ind[i]) if gi != i]
            # Jaccard re-rank
            pairs = []
            for gi, cos_sim in cand:
                if self.use_jaccard and self.camp_sets:
                    a = self.camp_sets.get(str(self.df.iloc[i]["product_id"]), set())
                    b = self.camp_sets.get(str(self.df.iloc[gi]["product_id"]), set())
                    jac = jaccard(a, b)
                    final = (1.0 - self.jaccard_weight) * cos_sim + self.jaccard_weight * jac
                else:
                    final = cos_sim
                pairs.append((gi, final))
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:k]

            row = {"product_id": pid, "product_name": pname}
            for rank, (gi, final) in enumerate(pairs, start=1):
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

@st.cache_data(show_spinner=False)
def prepare_prod_and_sets(raw_df, day_cols, store_code_or_none):
    prod_ = fe_time_series_per_product(raw_df, day_cols, store_filter=store_code_or_none)
    camp_ = build_campaign_sets(raw_df, store_filter=store_code_or_none)
    return prod_, camp_

# Model fit'i cache'le (aynı parametrelerle tekrar tekrar eğitme)
@st.cache_resource(show_spinner=False)
def fit_model(prod, camp_sets, alpha, beta, gamma, delta, use_j, jw):
    m = ProductSimilarityModel(alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                               use_jaccard=use_j, jaccard_weight=jw).fit(prod, campaign_sets=camp_sets)
    return m

uploaded = st.file_uploader("Excel yükleyin", type=["xlsx"])
if uploaded:
    raw, day_cols = load_campaign_excel(uploaded)
    st.success(f"{len(raw):,} satır yüklendi • Gün kolonları: {', '.join(day_cols[:3])} ...")

    # ---- Kapsam seçimi (sidebar) ----
    st.sidebar.markdown("### Kapsam")
    scope = st.sidebar.radio("Benzerlik kapsamı", ["Tüm mağazalar (agregat)", "Belirli mağaza"])
    selected_store = None
    if scope == "Belirli mağaza":
        store_options = raw["store_code"].astype(str).unique().tolist()
        selected_store = st.sidebar.selectbox("Mağaza (store_code)", sorted(store_options))

    # Store‑bağlamlı özet + kampanya kümeleri
    prod, camp_sets = prepare_prod_and_sets(raw, day_cols, selected_store)

    st.write(f"Ürün sayısı (bu kapsamda): {len(prod):,}")

    with st.expander("⚙️ Ağırlıklar / Ayarlar"):
        alpha = st.slider("Text (α)", 0.0, 2.0, ALPHA, 0.1)
        beta  = st.slider("Kategori (β)", 0.0, 2.0, BETA, 0.1)
        gamma = st.slider("Sayısal (γ)", 0.0, 2.0, GAMMA, 0.1)
        delta = st.slider("TS Şekil (δ)", 0.0, 2.0, DELTA, 0.1)
        use_j = st.checkbox("Kampanya Jaccard re‑rank", value=True)
        jw = st.slider("Jaccard ağırlığı (w)", 0.0, 0.8, 0.2, 0.05)

    model = fit_model(prod, camp_sets, alpha, beta, gamma, delta, use_j, jw)

    tab1, tab2, tab3 = st.tabs(["🔍 Tek ürün arama", "📋 Toplu benzerlik tablosu", "🗺 Harita (SVD)"])

    with tab1:
        options = [f"{row['product_id']} - {row['product_name']}" for _, row in prod.iterrows()]
        sel = st.selectbox("Ürün seç", options)
        sel_code = sel.split(" - ", 1)[0]

        # Ürün bu kapsamda yoksa uyarı
        if not (prod["product_id"] == sel_code).any():
            scope_txt = "seçili mağazada" if selected_store else "tüm veri kümesinde"
            st.warning(f"Seçilen ürün {scope_txt} bulunamadı.")
        else:
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
        # TruncatedSVD: Sparse matrisi dense'e çevirmeden 2D'ye indirger (RAM-safe)
        svd = TruncatedSVD(n_components=2, random_state=42)
        coords = svd.fit_transform(model.X_all)  # model.X_all sparse -> OK

        pdf = pd.DataFrame(coords, columns=["x","y"])
        pdf["product_id"] = prod["product_id"]
        pdf["product_name"] = prod["product_name"]

        sel2 = st.selectbox(
            "Haritada öne çıkar",
            [f"{c} - {n}" for c, n in zip(prod["product_id"], prod["product_name"])],
            key="map_sel"
        )
        sel_code2 = sel2.split(" - ", 1)[0]
        sel_idx_arr = np.where(prod["product_id"].astype(str).values == str(sel_code2))[0]
        if len(sel_idx_arr) == 0:
            st.warning("Seçilen ürün bulunamadı.")
        else:
            sel_idx = int(sel_idx_arr[0])
            nn = model.search_by_code(sel_code2, k=3)
            nn_idx = [gi for gi, _, _ in nn]

            fig = go.Figure()
            # Tüm noktalar (gri)
            fig.add_trace(go.Scatter(
                x=pdf["x"], y=pdf["y"], mode="markers",
                marker=dict(size=6, color="rgba(200,200,200,0.5)"),
                hoverinfo="skip", showlegend=False
            ))
            # Komşular (turuncu)
            if nn_idx:
                fig.add_trace(go.Scatter(
                    x=pdf.iloc[nn_idx]["x"], y=pdf.iloc[nn_idx]["y"],
                    mode="markers", marker=dict(size=10, color="#e96411"),
                    hoverinfo="skip", showlegend=False
                ))

            # Seçili ürün (mavi)
            sx, sy = float(pdf.iloc[sel_idx]["x"]), float(pdf.iloc[sel_idx]["y"])
            fig.add_trace(go.Scatter(
                x=[sx], y=[sy], mode="markers",
                marker=dict(size=12, color="#0D0DD4"),
                hoverinfo="skip", showlegend=False
            ))

            base = prod.iloc[sel_idx]
            sel_info = (
                f"<b>{base['product_id']}</b> — {base['product_name']}<br>"
                f"{base['brand']}<br>"
                f"{base['first_category']} > {base['second_category']} > "
                f"{base['third_category']} > {base['fourth_category']}"
            )
            fig.add_annotation(
                x=sx, y=sy, xanchor="left", yanchor="bottom",
                xshift=12, yshift=12, showarrow=False,
                text=sel_info, align="left",
                font=dict(size=12, color="#fff"),
                bgcolor="rgba(0,0,0,0.85)", bordercolor="#0D0DD4",
                borderwidth=1, borderpad=6
            )

            x_panel = pdf["x"].max() + (pdf["x"].max() - pdf["x"].min()) * 0.2
            y_top = pdf["y"].max()
            y_gap = 0.5 * (pdf["y"].max() - pdf["y"].min())

            for i, gi in enumerate(nn_idx, start=1):
                r = prod.iloc[gi]
                info = (
                    f"<b>{r['product_id']}</b> — {r['product_name']}<br>{r['brand']}<br>"
                    f"{r['first_category']} > {r['second_category']} > "
                    f"{r['third_category']} > {r['fourth_category']}"
                )
                fig.add_annotation(
                    x=x_panel, y=y_top - (i-1)*y_gap,
                    xanchor="left", yanchor="top",
                    text=info, showarrow=False, align="left",
                    font=dict(size=12, color="#fff"),
                    bgcolor="rgba(0,0,0,0.85)", bordercolor="#e96411",
                    borderwidth=1, borderpad=6
                )

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode=False
            )
            st.plotly_chart(fig, use_container_width=True)
