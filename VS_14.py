# VS_14.py
# ==========================================================
# Dashboard de Denuncias de Violencia sexual contra niñas menores de 15 años en Guatemala
# - 22 departamentos garantizados (tabla).
# - Gráficos Top N (por defecto 10).
# - Filtro "Departamento → ver municipios".
# - Enfoque en SENTENCIAS + tabla resumen para decisiones.
# - Acciones de incidencia (3 ideas) DINÁMICAS y con selector de enfoque.
# - Columna de ALERTA por territorio (rojo/ámbar/verde) + motivo.
# - Exportar FICHA (PNG / PDF) con KPIs + tabla + período de datos.
#   streamlit run VS_14.py
# ==========================================================

import inspect
import unicodedata
from io import BytesIO
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image, ImageDraw, ImageFont

# ================== CONFIG / TEMA ==================
st.set_page_config(
    page_title="Denuncias de Violencia sexual contra niñas menores de 15 años en Guatemala.",
    page_icon="🟣",
    layout="wide"
)

COLORS = {
    "bg": "#F9F7FB",
    "text": "#2E2459",
    "muted": "#5E5A78",
    "primary": "#6A1B9A",
    "primary2": "#8E24AA",
    "accent": "#FDD835",
    "accent2": "#FFB300",
    "line": "#D1C4E9",
}

st.markdown(f"""
<style>
html, body, [class*="stApp"] {{
  background: {COLORS['bg']};
  color: {COLORS['text']};
}}
h1, h2, h3, h4 {{ color: {COLORS['primary']}; }}
hr {{ border: none; border-top: 1px solid {COLORS['line']}; margin: .8rem 0 1.2rem 0; }}
.stButton>button {{ background: {COLORS['primary']}; color: white; border-radius: 999px; border: none; }}
.stDownloadButton>button {{ background: {COLORS['accent']}; color: #3a3200; border-radius: 999px; border: none; }}
.metric-small > div > div:nth-child(1) {{ font-size: .9rem; color: {COLORS['muted']}; }}
.metric-small > div > div:nth-child(2) {{ font-size: 1.2rem; color: {COLORS['primary']}; }}
</style>
""", unsafe_allow_html=True)

# ---------- Compatibilidad de ancho (width vs use_container_width) ----------
def _supports_width(func) -> bool:
    try:
        return "width" in inspect.signature(func).parameters
    except Exception:
        return False

_HAS_WIDTH_ALTAIR = _supports_width(st.altair_chart)
_HAS_WIDTH_DF = _supports_width(st.dataframe)

def ui_altair(chart):
    """Muestra gráficos Altair con compatibilidad hacia atrás."""
    if _HAS_WIDTH_ALTAIR:
        st.altair_chart(chart, width="stretch")
    else:
        st.altair_chart(chart, use_container_width=True)

def ui_dataframe(df, **kwargs):
    """Muestra dataframes con compatibilidad hacia atrás."""
    if _HAS_WIDTH_DF:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)

# ================== CATÁLOGO OFICIAL 22 DEPARTAMENTOS ==================
DEPARTAMENTOS_GT = [
    "Alta Verapaz","Baja Verapaz","Chimaltenango","Chiquimula","El Progreso","Escuintla",
    "Guatemala","Huehuetenango","Izabal","Jalapa","Jutiapa","Petén","Quetzaltenango",
    "Quiché","Retalhuleu","Sacatepéquez","San Marcos","Santa Rosa","Sololá",
    "Suchitepéquez","Totonicapán","Zacapa"
]
def _rm_accents(s): 
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")
def _canon_dep(s): 
    return _rm_accents(s).strip().lower()
CANON_DEP_MAP = {_canon_dep(d): d for d in DEPARTAMENTOS_GT}

# ================== DATOS ==================
DEFAULT_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS5TXNsUgBgMP6Cdv1Utkd01MkZi2Vmg7bbJBwkpbdL8jY-cwKP1WjtzHoaJDt4KcsPR2SpYPXBFUwo/pub?output=csv"

st.sidebar.header("Fuente de datos")
DATA_URL = st.sidebar.text_input("CSV/Google Sheet (público en formato CSV)", value=DEFAULT_CSV_URL)

@st.cache_data(show_spinner=False)
def cargar_datos(url: str) -> pd.DataFrame:
    import pandas as pd, unicodedata
    df = pd.read_csv(url)

    def norm_col(s):
        s = str(s).strip()
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        return s.lower()

    cols = {norm_col(c): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    col_fecha = pick("fecha denuncia", "fechadenuncia", "fecha", "fecha_hecho", "fechahecho")
    col_pais  = pick("pais")
    col_depto = pick("departamento", "depto", "dpto")
    col_muni  = pick("municipio", "muni")
    col_sexo  = pick("sexo", "genero")
    col_edad  = pick("edad")
    col_delito= pick("delito", "delitos", "tipo_delito", "tipo de delito")
    col_estado= pick("estado caso", "estadocaso", "estado", "estado_del_caso")
    col_resol = pick("resolucion", "resolución", "resultado", "sentencia")

    rename_map = {}
    if col_fecha: rename_map[col_fecha] = "fecha_denuncia"
    if col_pais:  rename_map[col_pais]  = "pais"
    if col_depto: rename_map[col_depto] = "departamento"
    if col_muni:  rename_map[col_muni]  = "municipio"
    if col_sexo:  rename_map[col_sexo]  = "sexo"
    if col_edad:  rename_map[col_edad]  = "edad"
    if col_delito:rename_map[col_delito]= "delito"
    if col_estado:rename_map[col_estado]= "estado_caso"
    if col_resol: rename_map[col_resol] = "resolucion"
    df = df.rename(columns=rename_map)

    if "fecha_denuncia" in df.columns:
        df["fecha_denuncia"] = pd.to_datetime(df["fecha_denuncia"], errors="coerce")
        df["anio"]       = df["fecha_denuncia"].dt.year
        df["mes"]        = df["fecha_denuncia"].dt.to_period("M").astype(str)
        df["trimestre"]  = df["fecha_denuncia"].dt.to_period("Q").astype(str)

    for c in ["pais","sexo","delito","estado_caso","resolucion"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "departamento" in df.columns:
        df["departamento"] = df["departamento"].astype(str).str.strip()
        df["dep_key"] = df["departamento"].map(_canon_dep)
        df["departamento_std"] = df["dep_key"].map(CANON_DEP_MAP).fillna(df["departamento"].str.title())
    if "municipio" in df.columns:
        df["municipio"] = df["municipio"].astype(str).str.strip().str.title()
    if "edad" in df.columns:
        df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    return df

try:
    df = cargar_datos(DATA_URL)
except Exception as e:
    st.error(f"Error al cargar datos desde la URL. Verifica que sea un CSV público. Detalle: {e}")
    st.stop()

# ================== HEADER ==================
st.title("Denuncias de Violencia sexual contra niñas menores de 15 años en Guatemala.")
st.caption("Análisis de violencia sexual por territorio, temporalidad, estados y delitos.")

# ================== SIDEBAR: FILTROS ==================
st.sidebar.header("Filtros")

nivel_base = st.sidebar.radio("Nivel de análisis", ["Departamento", "Municipio"], index=0)
depto_focus = st.sidebar.selectbox("Departamento → ver municipios", ["Todos"] + DEPARTAMENTOS_GT, index=0)

if "pais" in df.columns:
    paises = ["Todos"] + sorted([p for p in df["pais"].dropna().unique() if str(p).strip()])
    pais_sel = st.sidebar.selectbox("País", paises)
else:
    pais_sel = "Todos"

gran = st.sidebar.radio("Granularidad (sentencias)", ["Año", "Trimestre", "Mes"], index=2)
periodo_col = "anio" if gran == "Año" else ("trimestre" if gran == "Trimestre" else "mes")
top_n = st.sidebar.slider("Top N (solo gráficos)", 5, 15, 10, step=1)

enfoque_sel = st.sidebar.selectbox(
    "Enfoque de incidencia",
    ["Automático", "Prevención", "Protección", "Acceso a justicia", "Datos y transparencia"],
    index=0,
    help="Prioriza el tipo de acción en la redacción de recomendaciones."
)

# Nivel efectivo
nivel = "Municipio" if depto_focus != "Todos" else nivel_base
territorio_col = "departamento_std" if nivel == "Departamento" else "municipio"

# Opciones de territorios
if nivel == "Departamento":
    territorios_all = DEPARTAMENTOS_GT[:]
else:
    if depto_focus != "Todos":
        base_series = df.loc[df["departamento_std"] == depto_focus, "municipio"]
    else:
        base_series = df["municipio"]
    territorios_all = sorted([t for t in base_series.dropna().unique() if str(t).strip()])

sexos_all   = sorted([s for s in df.get("sexo", pd.Series(dtype=str)).dropna().unique() if str(s).strip()])
delitos_all = sorted([d for d in df.get("delito", pd.Series(dtype=str)).dropna().unique() if str(d).strip()])
estados_all = sorted([e for e in df.get("estado_caso", pd.Series(dtype=str)).dropna().unique() if str(e).strip()])
res_all     = sorted([r for r in df.get("resolucion", pd.Series(dtype=str)).dropna().unique() if str(r).strip()])

territorios_sel = st.sidebar.multiselect(f"{nivel}(s)", options=territorios_all, default=territorios_all)
sexos_sel   = st.sidebar.multiselect("Sexo", options=sexos_all, default=sexos_all)
delitos_sel = st.sidebar.multiselect("Delito", options=delitos_all, default=delitos_all[:10])
estados_sel = st.sidebar.multiselect("Estado del caso", options=estados_all, default=estados_all)
res_sel     = st.sidebar.multiselect("Resolución (opcional)", options=res_all, default=res_all)

# Rango de fechas
if "fecha_denuncia" in df.columns:
    fmin = pd.to_datetime(df["fecha_denuncia"], errors="coerce").min()
    fmax = pd.to_datetime(df["fecha_denuncia"], errors="coerce").max()
    if pd.notna(fmin) and pd.notna(fmax):
        f_ini, f_fin = st.sidebar.date_input("Rango de fechas",
                                             value=[fmin.date(), fmax.date()],
                                             min_value=fmin.date(), max_value=fmax.date())
    else:
        f_ini, f_fin = None, None
else:
    f_ini, f_fin = None, None

# ================== APLICAR FILTROS ==================
df_f = df.copy()
if pais_sel != "Todos" and "pais" in df_f.columns:
    df_f = df_f[df_f["pais"] == pais_sel]
if depto_focus != "Todos" and "departamento_std" in df_f.columns:
    df_f = df_f[df_f["departamento_std"] == depto_focus]
if territorios_sel:
    df_f = df_f[df_f[territorio_col].isin(territorios_sel)]
if sexos_sel and "sexo" in df_f.columns:
    df_f = df_f[df_f["sexo"].isin(sexos_sel)]
if delitos_sel and "delito" in df_f.columns and len(delitos_sel) > 0:
    df_f = df_f[df_f["delito"].isin(delitos_sel)]
if estados_sel and "estado_caso" in df_f.columns and len(estados_sel) > 0:
    df_f = df_f[df_f["estado_caso"].isin(estados_sel)]
if res_sel and "resolucion" in df_f.columns and len(res_sel) > 0:
    df_f = df_f[df_f["resolucion"].isin(res_sel)]
if f_ini and f_fin and "fecha_denuncia" in df_f.columns:
    df_f = df_f[(df_f["fecha_denuncia"] >= pd.to_datetime(f_ini)) &
                (df_f["fecha_denuncia"] <= pd.to_datetime(f_fin))]

# ================== SUBCONJUNTO SOLO SENTENCIAS ==================
def _lower(s): 
    return str(s).strip().lower() if pd.notna(s) else s

df_sent = df_f.copy()
if "estado_caso" in df_sent.columns:
    df_sent = df_sent[df_sent["estado_caso"].map(_lower) == "sentencia"]
else:
    df_sent = df_sent.iloc[0:0]

def _norm(s):
    if pd.isna(s): 
        return ""
    s = str(s).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def map_res(v):
    t = _norm(v)
    if not t:
        return None
    if "conden" in t:
        return "Condenatoria"
    if ("absolu" in t) or ("absuelt" in t) or ("absuelv" in t) or ("se absuelv" in t) or ("absuelve" in t):
        return "Absolutoria"
    return None

if "resolucion" in df_sent.columns:
    df_sent["resolucion_std"] = df_sent["resolucion"].map(map_res)

# ================== KPIs ==================
total_casos = len(df_f)
total_sent  = len(df_sent)
base_res    = df_sent.dropna(subset=["resolucion_std"]).shape[0] if "resolucion_std" in df_sent.columns else 0
n_condena   = (df_sent["resolucion_std"] == "Condenatoria").sum() if "resolucion_std" in df_sent.columns else 0
n_absol     = (df_sent["resolucion_std"] == "Absolutoria").sum() if "resolucion_std" in df_sent.columns else 0

def pct(a, b): 
    return (100.0 * a / b) if (b and b > 0) else 0.0
def fmt(x, nd=1):
    try: return f"{x:,.{nd}f}".replace(",", " ")
    except: return str(x)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Casos (filtros)", f"{total_casos:,}")
with c2:
    st.metric("Sentencias", f"{total_sent:,}")
with c3:
    st.metric("% condena (sobre sentencias con resolución)", f"{fmt(pct(n_condena, base_res))}%")
with c4:
    st.metric("% absolución (sobre sentencias con resolución)", f"{fmt(pct(n_absol, base_res))}%")

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== ESTADOS DE CASO (GRÁFICO SIMPLE) ==================
st.subheader("Estados de caso (distribución simple)")
if "estado_caso" in df_f.columns and not df_f.empty:
    est = (df_f["estado_caso"].dropna().value_counts().reset_index())
    est.columns = ["estado_caso", "n"]
    chart_estado_simple = alt.Chart(est).mark_bar().encode(
        x=alt.X("n:Q", title="Cantidad"),
        y=alt.Y("estado_caso:N", sort='-x', title="Estado"),
        color=alt.value(COLORS["primary"]),
        tooltip=["estado_caso", "n"]
    ).properties(height=300)
    ui_altair(chart_estado_simple)
else:
    st.info("No hay datos de estados de caso para este filtro.")

# ================== HELPERS ==================
def ensure_22_departments(df_grouped: pd.DataFrame, dep_col: str = "departamento_std", count_col: str = "n") -> pd.DataFrame:
    base = pd.DataFrame({dep_col: DEPARTAMENTOS_GT})
    out = base.merge(df_grouped, on=dep_col, how="left")
    if count_col not in out.columns:
        other_cols = [c for c in out.columns if c != dep_col]
        if other_cols:
            out = out.rename(columns={other_cols[0]: count_col})
        else:
            out[count_col] = 0
    out[count_col] = out[count_col].fillna(0).astype(int)
    return out

def _state_shares_by_territory(df_all: pd.DataFrame, terr_col: str) -> pd.DataFrame:
    if ("estado_caso" not in df_all.columns) or (terr_col not in df_all.columns) or df_all.empty:
        return pd.DataFrame(columns=[terr_col, "share_archivo", "share_tramite"]).set_index(terr_col)
    tmp = df_all[[terr_col, "estado_caso"]].dropna()
    tmp["estado_norm"] = tmp["estado_caso"].astype(str).str.lower()
    tmp["cat"] = "otros"
    tmp.loc[tmp["estado_norm"].str.contains(r"archiv|desist|sobrese", regex=True), "cat"] = "archivo"
    tmp.loc[tmp["estado_norm"].str.contains(r"tr[aá]mite|investig", regex=True), "cat"] = "tramite"
    counts = tmp.groupby([terr_col, "cat"]).size().unstack(fill_value=0)
    for col in ["archivo", "tramite"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["total_est"] = counts.sum(axis=1).replace(0, pd.NA)
    counts["share_archivo"] = (counts["archivo"] / counts["total_est"]).fillna(0.0)
    counts["share_tramite"] = (counts["tramite"] / counts["total_est"]).fillna(0.0)
    return counts[["share_archivo", "share_tramite"]]

def _pct_condena_by_territory(df_s: pd.DataFrame, terr_col: str) -> pd.DataFrame:
    if df_s.empty or ("resolucion_std" not in df_s.columns) or (terr_col not in df_s.columns):
        return pd.DataFrame(columns=[terr_col, "total_sent", "condena", "pct_condena"]).set_index(terr_col)
    base = (df_s.dropna(subset=[terr_col, "resolucion_std"])
                .groupby([terr_col, "resolucion_std"]).size().reset_index(name="n"))
    tot = base.groupby(terr_col)["n"].sum().rename("total_sent")
    conde = base[base["resolucion_std"]=="Condenatoria"].groupby(terr_col)["n"].sum().rename("condena")
    met = pd.concat([tot, conde], axis=1).fillna(0)
    met["pct_condena"] = (met["condena"] / met["total_sent"] * 100).replace([float("inf")], 0).fillna(0.0)
    return met

def _classify_alert_row(pct_condena, share_archivo, share_tramite, nat_median, sent_min_ok) -> tuple:
    motivos = []
    if nat_median is None:
        nat_median = pct_condena
    if sent_min_ok and (pct_condena < nat_median - 10) and (share_archivo > 0.20):
        if pct_condena < nat_median - 10:
            motivos.append(f"% condena por debajo de mediana nacional ({nat_median:.1f}%)")
        if share_archivo > 0.20:
            motivos.append("alto archivo/desistimiento (>20%)")
        return "Alta", "; ".join(motivos) if motivos else "riesgo elevado"
    cond_media = []
    if pct_condena < nat_median - 5:
        cond_media.append("% condena bajo referencia")
    if share_tramite > 0.30:
        cond_media.append("mora en trámite/investigación (>30%)")
    if share_archivo > 0.15:
        cond_media.append("tasa de archivo relevante (>15%)")
    if cond_media:
        return "Media", "; ".join(cond_media)
    return "Baja", "sin señales críticas"

# ====== TABLA RESUMEN (Denuncias / Sentencias / Tasas / ALERTA) ======
def build_summary_table(df_all: pd.DataFrame, df_s: pd.DataFrame, terr_col: str, nivel_txt: str) -> pd.DataFrame:
    den = df_all.groupby(terr_col).size().rename("Denuncias")
    sen = df_s.groupby(terr_col).size().rename("Sentencias") if not df_s.empty else pd.Series(dtype=int, name="Sentencias")

    if not df_s.empty and "resolucion_std" in df_s.columns:
        conde = df_s[df_s["resolucion_std"] == "Condenatoria"].groupby(terr_col).size().rename("Condenatorias")
        absol = df_s[df_s["resolucion_std"] == "Absolutoria"].groupby(terr_col).size().rename("Absoluciones")
    else:
        conde = pd.Series(dtype=int, name="Condenatorias")
        absol = pd.Series(dtype=int, name="Absoluciones")

    tabla = pd.concat([den, sen, conde, absol], axis=1).fillna(0)

    # Conteos a int
    for c in ["Denuncias", "Sentencias", "Condenatorias", "Absoluciones"]:
        if c not in tabla.columns:
            tabla[c] = 0
        tabla[c] = tabla[c].fillna(0).astype(int)

    shares = _state_shares_by_territory(df_all, terr_col)
    met_sent = _pct_condena_by_territory(df_s, terr_col)

    tabla = tabla.join(shares, how="left").join(
        met_sent[["pct_condena", "total_sent"]] if not met_sent.empty else pd.DataFrame(index=tabla.index), how="left"
    )
    for c in ["share_archivo", "share_tramite", "pct_condena"]:
        if c not in tabla.columns:
            tabla[c] = 0.0
        tabla[c] = tabla[c].fillna(0.0).astype(float)
    if "total_sent" not in tabla.columns:
        tabla["total_sent"] = 0
    tabla["total_sent"] = tabla["total_sent"].fillna(0).astype(int)

    tabla["Tasa de sentencia (%)"] = ((tabla["Sentencias"] / tabla["Denuncias"]).replace([float("inf")], 0).fillna(0) * 100)
    tabla["% condena"] = ((tabla["Condenatorias"] / tabla["Sentencias"]).replace([float("inf")], 0).fillna(0) * 100)
    tabla["% absolución"] = ((tabla["Absoluciones"] / tabla["Sentencias"]).replace([float("inf")], 0).fillna(0) * 100)

    if nivel_txt == "Departamento":
        base = pd.DataFrame({"Departamento": DEPARTAMENTOS_GT}).set_index("Departamento")
        tabla.index.name = "Departamento"
        tabla = base.join(tabla, how="left")
        fill_map = {
            "Denuncias": 0, "Sentencias": 0, "Condenatorias": 0, "Absoluciones": 0,
            "Tasa de sentencia (%)": 0.0, "% condena": 0.0, "% absolución": 0.0,
            "share_archivo": 0.0, "share_tramite": 0.0, "pct_condena": 0.0, "total_sent": 0
        }
        tabla = tabla.fillna(fill_map)
        for c in ["Denuncias", "Sentencias", "Condenatorias", "Absoluciones", "total_sent"]:
            tabla[c] = tabla[c].astype(int)

    sent_min = max(5, round(0.01 * int(tabla["total_sent"].sum() or 0)))
    if (tabla["total_sent"] >= sent_min).any():
        nat_median = float(tabla.loc[tabla["total_sent"] >= sent_min, "pct_condena"].median())
    else:
        nat_median = None

    etiquetas, motivos = [], []
    for _, r in tabla.iterrows():
        ts_val = r.get("total_sent", 0)
        try:
            ts_int = int(ts_val) if pd.notna(ts_val) else 0
        except Exception:
            ts_int = 0
        et, mo = _classify_alert_row(
            pct_condena=float(r.get("pct_condena", 0.0) or 0.0),
            share_archivo=float(r.get("share_archivo", 0.0) or 0.0),
            share_tramite=float(r.get("share_tramite", 0.0) or 0.0),
            nat_median=nat_median,
            sent_min_ok=bool(ts_int >= sent_min)
        )
        etiquetas.append(et); motivos.append(mo)
    tabla["Alerta"] = etiquetas
    tabla["Motivo alerta"] = motivos

    tabla = tabla.reset_index().rename(columns={terr_col: nivel_txt})
    tabla = tabla.sort_values(["Denuncias", "Sentencias"], ascending=False)
    return tabla

# --------- Render de FICHA (PNG / PDF) -------------
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def make_ficha_image(
    titulo: str,
    subtitulo: str,
    kpi_dict: dict,
    tabla_sum: pd.DataFrame,
    nivel_txt: str,
    periodo_txt: str,
    width: int = 1600,
    margin: int = 60,
) -> Image.Image:
    bg = (249, 247, 251)
    text = (46, 36, 89)
    primary = (106, 27, 154)
    line = (209, 196, 233)

    rows = min(len(tabla_sum), 12)
    height = margin*2 + 180 + 140 + (rows+2)*45 + 80
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    f_title = _load_font(40)
    f_sub = _load_font(26)
    f_k = _load_font(28)
    f_kv = _load_font(36)
    f_th = _load_font(24)
    f_td = _load_font(22)
    f_foot = _load_font(20)

    y = margin
    draw.text((margin, y), titulo, fill=primary, font=f_title)
    y += 56
    draw.text((margin, y), subtitulo, fill=text, font=f_sub)
    y += 50
    draw.line([(margin, y), (width-margin, y)], fill=line, width=2)
    y += 20

    kpi_w = (width - 2*margin) // 4
    k_labels = list(kpi_dict.keys())
    for i, k in enumerate(k_labels):
        x0 = margin + i*kpi_w
        draw.text((x0, y), k, fill=text, font=f_k)
        draw.text((x0, y+34), str(kpi_dict[k]), fill=primary, font=f_kv)
    y += 90

    draw.text((margin, y), f"Tabla resumen por {nivel_txt} (máx. 12 filas)", fill=text, font=f_th)
    y += 36

    cols = [nivel_txt, "Denuncias", "Sentencias", "Tasa de sentencia (%)", "% condena", "% absolución"]
    tbl = tabla_sum[cols].head(12).copy()

    col_w = [0.34, 0.13, 0.13, 0.16, 0.12, 0.12]
    start_x = margin
    usable_w = width - 2*margin

    for cx, _ in enumerate(col_w):
        x = start_x + int(sum(col_w[:cx]) * usable_w)
        draw.text((x, y), cols[cx], fill=primary, font=f_th)
    y += 32
    draw.line([(margin, y), (width-margin, y)], fill=line, width=1)
    y += 10

    for _, row in tbl.iterrows():
        for cx, _ in enumerate(col_w):
            x = start_x + int(sum(col_w[:cx]) * usable_w)
            val = row[cols[cx]]
            if isinstance(val, float):
                txt = f"{val:.1f}%" if ("%" in cols[cx] or "sentencia" in cols[cx].lower()) else f"{val:.0f}"
            else:
                txt = str(val)
            draw.text((x, y), txt, fill=text, font=f_td)
        y += 36

    y += 8
    draw.line([(margin, y), (width-margin, y)], fill=line, width=1)
    y += 16

    draw.text((margin, y), f"Período de datos: {periodo_txt}", fill=text, font=f_foot)
    return img

# ================== RESOLUCIONES POR TERRITORIO (SOLO SENTENCIAS) ==================
st.subheader(f"Resoluciones por {('Departamento' if nivel=='Departamento' else 'Municipio')} (solo sentencias, distribución 100%)")
if not df_sent.empty and "resolucion_std" in df_sent.columns and territorio_col in df_sent.columns:
    base = (df_sent.dropna(subset=[territorio_col, "resolucion_std"])
                 .groupby([territorio_col, "resolucion_std"]).size().reset_index(name="n"))

    if nivel == "Departamento":
        tot_dep = (base.groupby(territorio_col)["n"].sum()
                        .reindex(DEPARTAMENTOS_GT)
                        .fillna(0).astype(int))
        tabla_dep = ensure_22_departments(tot_dep.reset_index(name="n"),
                                          dep_col=territorio_col, count_col="n").sort_values("n", ascending=False)
        top_names = tabla_dep.head(top_n)[territorio_col].tolist()
        base_plot = base[base[territorio_col].isin(top_names)].copy()
        orden = top_names
    else:
        tot_mun = base.groupby(territorio_col)["n"].sum().sort_values(ascending=False)
        top_names = tot_mun.head(top_n).index.tolist()
        base_plot = base[base[territorio_col].isin(top_names)].copy()
        orden = top_names

    tot = base_plot.groupby(territorio_col)["n"].transform("sum")
    base_plot["pct"] = (base_plot["n"] / tot) * 100

    chart_stack = alt.Chart(base_plot).mark_bar().encode(
        x=alt.X(f"{territorio_col}:N", sort=orden, title=("Departamento" if nivel=="Departamento" else "Municipio")),
        y=alt.Y("pct:Q", stack="normalize", title="Porcentaje"),
        color=alt.Color("resolucion_std:N", title="Resolución",
                        scale=alt.Scale(range=[COLORS["primary"], COLORS["accent2"], "#bbb"])),
        tooltip=[territorio_col, "resolucion_std",
                 alt.Tooltip("n:Q", title="Cantidad"),
                 alt.Tooltip("pct:Q", format=".1f", title="%")]
    ).properties(height=380)
    ui_altair(chart_stack)

    # ==== Tabla resumen (Denuncias + Sentencias + Tasas + ALERTA) ====
    nivel_txt = "Departamento" if nivel=="Departamento" else "Municipio"
    tabla_sum = build_summary_table(df_f, df_sent, territorio_col, nivel_txt)

    st.markdown(f"#### Tabla resumen por {nivel_txt}")
    activar_calor = st.checkbox("Activar color de calor en celdas", value=True,
                                help="Aplica degradado a columnas de volumen y desempeño, y color por alerta.")

    # Descarga CSV
    csv_bytes = tabla_sum.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV (tabla resumen)", data=csv_bytes,
                       file_name=f"tabla_resumen_{nivel_txt.lower()}.csv", mime="text/csv")

    # Dataframe configurado
    try:
        max_den = int(tabla_sum["Denuncias"].max() or 0)
        max_sen = int(tabla_sum["Sentencias"].max() or 0)
        ui_dataframe(
            tabla_sum,
            column_config={
                nivel_txt: st.column_config.TextColumn(nivel_txt),
                "Denuncias": st.column_config.ProgressColumn("Denuncias", min_value=0, max_value=max_den, format="%d"),
                "Sentencias": st.column_config.ProgressColumn("Sentencias", min_value=0, max_value=max_sen, format="%d"),
                "Tasa de sentencia (%)": st.column_config.ProgressColumn("Tasa de sentencia (%)", min_value=0.0, max_value=100.0, format="%.1f%%"),
                "% condena": st.column_config.ProgressColumn("% condena", min_value=0.0, max_value=100.0, format="%.1f%%"),
                "% absolución": st.column_config.ProgressColumn("% absolución", min_value=0.0, max_value=100.0, format="%.1f%%"),
                "Condenatorias": st.column_config.NumberColumn("Condenatorias", format="%d"),
                "Absoluciones": st.column_config.NumberColumn("Absoluciones", format="%d"),
                "Alerta": st.column_config.TextColumn("Alerta"),
                "Motivo alerta": st.column_config.TextColumn("Motivo alerta"),
            }
        )
        if activar_calor:
            def _alert_color(val):
                if val == "Alta":  return "background-color: #ffd9d9"
                if val == "Media": return "background-color: #fff3cd"
                return "background-color: #d4edda"
            styled = (tabla_sum.style
                        .background_gradient(cmap="Purples", subset=["Denuncias","Sentencias"])
                        .background_gradient(cmap="Oranges", subset=["Tasa de sentencia (%)","% condena","% absolución"])
                        .map(_alert_color, subset=["Alerta"])
                        .format({
                            "Tasa de sentencia (%)":"{:.1f}%",
                            "% condena":"{:.1f}%",
                            "% absolución":"{:.1f}%"
                        }))
            st.write(styled)
    except Exception:
        styled = (tabla_sum.style
                    .background_gradient(cmap="Purples", subset=["Denuncias","Sentencias"])
                    .background_gradient(cmap="Oranges", subset=["Tasa de sentencia (%)","% condena","% absolución"])
                    .format({
                        "Tasa de sentencia (%)":"{:.1f}%",
                        "% condena":"{:.1f}%",
                        "% absolución":"{:.1f}%"
                    }))
        st.write(styled)

    # ====== EXPORTAR FICHA (PNG / PDF) ======
    titulo_ficha = "Denuncias de Violencia sexual contra niñas menores de 15 años en Guatemala."
    subtitulo_ficha = f"Vista por {nivel_txt}" + (f" — {depto_focus}" if (nivel_txt=='Municipio' and depto_focus!='Todos') else "")

    kpis_ficha = {
        "Casos (filtros)": f"{total_casos:,}",
        "Sentencias": f"{total_sent:,}",
        "% condena": f"{pct(n_condena, base_res):.1f}%" if base_res > 0 else "0.0%",
        "% absolución": f"{pct(n_absol, base_res):.1f}%" if base_res > 0 else "0.0%",
    }

    if "fecha_denuncia" in df_f.columns and df_f["fecha_denuncia"].notna().any():
        pmin = pd.to_datetime(df_f["fecha_denuncia"]).min().date()
        pmax = pd.to_datetime(df_f["fecha_denuncia"]).max().date()
        periodo_txt = f"{pmin} a {pmax}"
    else:
        periodo_txt = "N/D"

    img = make_ficha_image(
        titulo=titulo_ficha,
        subtitulo=subtitulo_ficha,
        kpi_dict=kpis_ficha,
        tabla_sum=tabla_sum,
        nivel_txt=nivel_txt,
        periodo_txt=periodo_txt
    )

    buf_png = BytesIO(); img.save(buf_png, format="PNG"); png_bytes = buf_png.getvalue()
    buf_pdf = BytesIO(); img.save(buf_pdf, format="PDF"); pdf_bytes = buf_pdf.getvalue()

    cA, cB = st.columns(2)
    with cA:
        st.download_button("🖼️ Descargar FICHA (PNG)", data=png_bytes, file_name="ficha_vs15.png", mime="image/png")
    with cB:
        st.download_button("📄 Descargar FICHA (PDF)", data=pdf_bytes, file_name="ficha_vs15.pdf", mime="application/pdf")

else:
    st.info("No hay datos de sentencias (o resoluciones) para este filtro.")

# ================== TOP TERRITORIOS (CONDE/ABSUE) SOBRE SENTENCIAS ==================
st.subheader(f"¿Dónde se condena y se absuelve más? ({'Departamento' if nivel=='Departamento' else 'Municipio'}, sobre sentencias)")
if not df_sent.empty and "resolucion_std" in df_sent.columns and territorio_col in df_sent.columns:
    df_res = df_sent.dropna(subset=[territorio_col, "resolucion_std"])
    pivot = (df_res
             .groupby(territorio_col)["resolucion_std"].value_counts().unstack(fill_value=0)
             .rename(columns={"Condenatoria":"Condenatoria","Absolutoria":"Absolutoria"}))
    for col in ["Condenatoria","Absolutoria"]:
        if col not in pivot: pivot[col] = 0
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot[pivot["total"] > 0].copy()
    pivot["pct_condena"] = pivot["Condenatoria"] / pivot["total"] * 100.0
    pivot["pct_absol"]   = pivot["Absolutoria"]  / pivot["total"] * 100.0
    pivot = pivot.reset_index()
    pivot = pivot.sort_values("total", ascending=False).head(top_n)

    cL, cR = st.columns(2)
    with cL:
        st.caption("▲ Mayor % de condena (sobre sentencias)")
        top_condena = pivot.sort_values("pct_condena", ascending=False).head(top_n)
        chart_c = alt.Chart(top_condena).mark_bar().encode(
            x=alt.X("pct_condena:Q", title="% condena", axis=alt.Axis(format=".1f")),
            y=alt.Y(f"{territorio_col}:N", sort='-x', title=("Departamento" if nivel=="Departamento" else "Municipio")),
            color=alt.value(COLORS["primary"]),
            tooltip=[territorio_col, alt.Tooltip("total:Q", title="Sentencias"),
                     alt.Tooltip("pct_condena:Q", title="% condena", format=".1f")]
        ).properties(height=340)
        ui_altair(chart_c)

    with cR:
        st.caption("▼ Mayor % de absolución (sobre sentencias)")
        top_absol = pivot.sort_values("pct_absol", ascending=False).head(top_n)
        chart_a = alt.Chart(top_absol).mark_bar().encode(
            x=alt.X("pct_absol:Q", title="% absolución", axis=alt.Axis(format=".1f")),
            y=alt.Y(f"{territorio_col}:N", sort='-x', title=("Departamento" if nivel=="Departamento" else "Municipio")),
            color=alt.value(COLORS["accent2"]),
            tooltip=[territorio_col, alt.Tooltip("total:Q", title="Sentencias"),
                     alt.Tooltip("pct_absol:Q", title="% absolución", format=".1f")]
        ).properties(height=340)
        ui_altair(chart_a)
else:
    st.info("No hay suficientes sentencias para calcular rankings por territorio.")

# ================== SERIE TEMPORAL (SOLO SENTENCIAS) ==================
st.subheader(f"Serie temporal de sentencias por {gran}")
if not df_sent.empty:
    if periodo_col not in df_sent.columns:
        st.info("No se puede construir la serie temporal con las columnas disponibles.")
    else:
        total_t = df_sent.groupby([periodo_col]).size().reset_index(name="n_sent")
        if "resolucion_std" in df_sent.columns:
            by_res = df_sent.groupby([periodo_col, "resolucion_std"]).size().reset_index(name="n_res")

            line_total = alt.Chart(total_t).mark_line(point=True, stroke=COLORS["primary"]).encode(
                x=alt.X(f"{periodo_col}:O", title=gran),
                y=alt.Y("n_sent:Q", title="Sentencias"),
                tooltip=[alt.Tooltip(f"{periodo_col}:O", title=gran), alt.Tooltip("n_sent:Q", title="Sentencias")]
            )

            line_res = alt.Chart(by_res).mark_line(point=True).encode(
                x=alt.X(f"{periodo_col}:O", title=gran),
                y=alt.Y("n_res:Q", title="Sentencias"),
                color=alt.Color("resolucion_std:N", title="Resolución",
                                scale=alt.Scale(range=[COLORS["accent2"], COLORS["primary2"], "#777"])),
                tooltip=[alt.Tooltip(f"{periodo_col}:O", title=gran), "resolucion_std", alt.Tooltip("n_res:Q", title="Sentencias")]
            )
            ui_altair((line_total + line_res).properties(height=380))
        else:
            line_total = alt.Chart(total_t).mark_line(point=True, stroke=COLORS["primary"]).encode(
                x=alt.X(f"{periodo_col}:O", title=gran),
                y=alt.Y("n_sent:Q", title="Sentencias"),
                tooltip=[alt.Tooltip(f"{periodo_col}:O", title=gran), alt.Tooltip("n_sent:Q", title="Sentencias")]
            ).properties(height=380)
            ui_altair(line_total)
else:
    st.info("No hay datos de sentencias para este filtro.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== DELITOS (GENERAL, NO SOLO SENTENCIAS) ==================
st.subheader("Delitos más frecuentes (en el universo filtrado)")
if "delito" in df_f.columns and not df_f.empty:
    topn_del = (df_f.dropna(subset=["delito"])
                  .groupby("delito").size().reset_index(name="n")
                  .sort_values("n", ascending=False).head(15))
    chart_delito = alt.Chart(topn_del).mark_bar().encode(
        x=alt.X("n:Q", title="Cantidad"),
        y=alt.Y("delito:N", sort='-x', title="Delito"),
        color=alt.value(COLORS["primary"]),
        tooltip=["delito", "n"]
    ).properties(height=380)
    ui_altair(chart_delito)
else:
    st.info("No hay datos de delitos para este filtro.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== PROSA ENRIQUECIDA (CONCLUSIONES) ==================
st.subheader("Conclusiones")
def conclusiones(df_all: pd.DataFrame, df_s: pd.DataFrame, nivel_txt: str, periodo_txt: str) -> str:
    if df_all.empty:
        return "Con los filtros actuales no hay registros. Ajusta territorio, fechas o categorías."
    frases = []
    if "fecha_denuncia" in df_all.columns and df_all["fecha_denuncia"].notna().any():
        fmin = pd.to_datetime(df_all["fecha_denuncia"], errors="coerce").min()
        fmax = pd.to_datetime(df_all["fecha_denuncia"], errors="coerce").max()
        if pd.notna(fmin) and pd.notna(fmax):
            frases.append(f"El universo filtrado abarca del **{fmin.date()}** al **{fmax.date()}**.")
    frases.append(f"Casos totales analizados: **{len(df_all):,}**. Sentencias: **{len(df_s):,}**.")
    if not df_s.empty and "resolucion_std" in df_s.columns:
        base = df_s.dropna(subset=["resolucion_std"])
        n_res = len(base)
        if n_res > 0:
            n_c = (base["resolucion_std"] == "Condenatoria").sum()
            n_a = (base["resolucion_std"] == "Absolutoria").sum()
            frases.append(
                f"Sobre sentencias con resolución (**{n_res:,}**), **{(n_c/n_res*100):.1f}%** fueron *condenatorias* "
                f"y **{(n_a/n_res*100):.1f}%** *absolutorias*."
            )
            terr = (base.groupby(territorio_col)["resolucion_std"].value_counts().unstack(fill_value=0))
            for col in ["Condenatoria","Absolutoria"]:
                if col not in terr: terr[col] = 0
            terr["total"] = terr.sum(axis=1)
            terr = terr[terr["total"] >= max(5, round(0.01 * n_res))]
            if not terr.empty:
                terr["%Condena"] = terr["Condenatoria"]/terr["total"]*100
                terr["%Absolución"] = terr["Absolutoria"]/terr["total"]*100
                top_c = terr["%Condena"].sort_values(ascending=False).head(3)
                top_a = terr["%Absolución"].sort_values(ascending=False).head(3)
                if len(top_c) > 0:
                    frases.append("Mayor proporción de *condena* en " +
                                  ", ".join([f"**{idx}** ({val:.1f}%)" for idx, val in top_c.items()]) + ".")
                if len(top_a) > 0:
                    frases.append("Mayor proporción de *absolución* en " +
                                  ", ".join([f"**{idx}** ({val:.1f}%)" for idx, val in top_a.items()]) + ".")
    if "estado_caso" in df_all.columns:
        vc = df_all["estado_caso"].dropna().value_counts(normalize=True)*100
        if not vc.empty:
            frases.append("Distribución de estados (top): " +
                          ", ".join([f"**{k}** ({v:.1f}%)" for k, v in vc.head(3).items()]) + ".")
    if "delito" in df_all.columns:
        vd = df_all["delito"].dropna().value_counts().head(3)
        if not vd.empty:
            frases.append("Delitos más frecuentes: " +
                          ", ".join([f"**{k}** ({v})" for k, v in vd.items()]) + ".")
    if not df_s.empty and periodo_col in df_s.columns:
        serie = df_s.dropna(subset=[periodo_col]).groupby(periodo_col).size().sort_index()
        if len(serie) >= 4:
            q = max(1, len(serie)//4)
            ult = serie.tail(q).sum()
            ant = serie.iloc[:-q].tail(q).sum() if len(serie) >= 2*q else serie.iloc[:-q].sum()
            if ant > 0:
                var = (ult - ant) / ant * 100
                frases.append(f"En los {periodo_txt.lower()}s más recientes, las sentencias variaron **{var:.1f}%** vs. el periodo previo.")
    return " ".join(frases)

st.info(conclusiones(df_f, df_sent, nivel, gran))

# ================== ACCIONES DE INCIDENCIA (3 ideas) — dinámicas + enfoque ==================
st.subheader("Acciones de incidencia (3 ideas)")
def acciones_incidencia(df_all: pd.DataFrame, df_s: pd.DataFrame, enfoque: str = "Automático") -> list:
    ideas = []
    ambito = "departamentos" if territorio_col == "departamento_std" else "municipios"

    # Cuellos de botella globales
    share_archivo = share_tramite = 0.0
    if "estado_caso" in df_all.columns and not df_all.empty:
        estados = df_all["estado_caso"].astype(str).str.lower()
        share_archivo = estados.str.contains(r"archiv|desist|sobrese", regex=True).mean()
        share_tramite = estados.str.contains(r"tr[aá]mite|investig", regex=True).mean()

    # Baja % condena por territorio (si hay sentencias)
    foco_baja_condena = []
    nat_pct_condena = None
    if (not df_s.empty) and ("resolucion_std" in df_s.columns) and (territorio_col in df_s.columns):
        base = (df_s.dropna(subset=[territorio_col, "resolucion_std"])
                    .groupby([territorio_col, "resolucion_std"]).size().reset_index(name="n"))
        tot = base.groupby(territorio_col)["n"].sum().reset_index(name="total")
        conde = base[base["resolucion_std"]=="Condenatoria"].groupby(territorio_col)["n"].sum().reset_index(name="condena")
        met = tot.merge(conde, on=territorio_col, how="left").fillna(0)
        met["pct_condena"] = (met["condena"] / met["total"] * 100).fillna(0)

        if met["pct_condena"].notna().any():
            nat_pct_condena = float(met["pct_condena"].median())

        min_total = max(5, round(0.01 * met["total"].sum()))
        if nat_pct_condena is not None:
            crit = met[(met["total"] >= min_total) & (met["pct_condena"] < (nat_pct_condena - 10))]
            if crit.empty:
                crit = met[met["total"] >= min_total].sort_values(["pct_condena", "total"], ascending=[True, False]).head(3)
            foco_baja_condena = crit.sort_values(["pct_condena", "total"], ascending=[True, False]).head(3)[territorio_col].tolist()

    # Delitos predominantes
    top_del = []
    if "delito" in df_all.columns and not df_all.empty:
        top_del = df_all["delito"].dropna().astype(str).str.strip().value_counts().head(2).index.tolist()

    # Plantillas
    if foco_baja_condena:
        idea_justicia = (
            f"**Acceso a justicia con metas verificables**: instalar mesas MP–OJ–PGN en {ambito} **{', '.join(foco_baja_condena)}** "
            f"para acordar plazos de trámite, notificaciones y medidas de protección, con **metas de incremento de % de condena** "
            f"frente a la mediana nacional y seguimiento público en el tablero."
        )
    elif not df_s.empty:
        idea_justicia = (
            "**Fortalecimiento de ruta procesal**: estandarizar checklist de evidencias y peritajes sensibles a NNA, "
            "mejorar coordinación MP–OJ–PGN y publicar tiempos de cada hito (admisión, audiencia, sentencia) para reducir variabilidad territorial."
        )
    else:
        idea_justicia = (
            "**Primera respuesta y patrocinio temprano**: activar defensoría especializada y protocolos de recolección de evidencia, "
            "acompañamiento psicosocial, y derivación segura para disminuir desistimientos y llegar a sentencia."
        )

    if share_archivo >= max(share_tramite, 0.20):
        idea_cuellos = (
            "**Reducción de archivo/desistimiento**: fortalecer **patrocinio legal y apoyo psicosocial** a NNA y cuidadores, "
            "con alertas tempranas ante riesgos de revictimización; seguimiento de **tasa de archivo** por territorio en el tablero."
        )
    elif share_tramite > 0.30:
        idea_cuellos = (
            "**Gestión de mora en investigación/trámite**: acuerdos de gestión con fiscalías de niñez para acotar tiempos de investigación, "
            "agenda prioritaria de audiencias y tableros de **tiempos objetivo** por territorio."
        )
    else:
        idea_cuellos = (
            "**Cierre de brechas operativas**: auditorías rápidas de tiempos por etapa (denuncia–investigación–audiencia–sentencia) y "
            "alertas automáticas cuando se exceden plazos definidos interinstitucionalmente."
        )

    if top_del:
        idea_prev = (
            f"**Prevención comunitaria focalizada**: campañas con Mineduc y servicios de salud sobre **{', '.join(top_del)}**, "
            "material educativo en lenguas locales, canales de denuncia seguros y formación a personal docente; "
            "monitoreo de cobertura e impacto en el tablero."
        )
    else:
        idea_prev = (
            "**Prevención y detección temprana**: formación para escuelas y puestos de salud en señales de abuso, "
            "con rutas de derivación y protección visibles para NNA y cuidadores."
        )

    idea_datos = (
        "**Datos y transparencia**: mejorar diccionarios y metadatos, depurar inconsistencias territoriales, "
        "y publicar versiones de corte con cambios visibles; agregar indicadores de oportunidad (tiempos por etapa) al tablero."
    )

    base_ideas = [idea_justicia, idea_cuellos, idea_prev]

    if enfoque == "Acceso a justicia":
        ordered = [idea_justicia, idea_cuellos, idea_prev]
    elif enfoque == "Protección":
        ordered = [idea_cuellos, idea_justicia, idea_prev]
    elif enfoque == "Prevención":
        ordered = [idea_prev, idea_justicia, idea_cuellos]
    elif enfoque == "Datos y transparencia":
        ordered = [idea_datos, idea_justicia, idea_cuellos]
    else:
        ordered = base_ideas

    return ordered[:3]

for i, idea in enumerate(acciones_incidencia(df_f, df_sent, enfoque_sel), start=1):
    st.markdown(f"**{i}.** {idea}")

st.markdown("<hr/>", unsafe_allow_html=True)

# ================== TABLA DE FILA A FILA (datos crudos filtrados) ==================
st.subheader("Datos filtrados")
ui_dataframe(df_f)

# ================== PERÍODO DE DATOS (AL FINAL) ==================
if "fecha_denuncia" in df_f.columns and df_f["fecha_denuncia"].notna().any():
    pmin = pd.to_datetime(df_f["fecha_denuncia"]).min().date()
    pmax = pd.to_datetime(df_f["fecha_denuncia"]).max().date()
    st.markdown(f"**Período de datos:** {pmin} a {pmax}")
