"""
Dashboard principal del Sistema de Recomendacion Academica UTP.
Aplicacion Streamlit interactiva para docentes y asesores academicos.

Inicio rapido:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.ingestion import load_all_data
from pipeline.preprocessing import run_preprocessing
from pipeline.feature_engineering import create_features
from models.clustering import (
    train_clustering, assign_cluster_labels,
    get_cluster_statistics, CLUSTER_CONFIG
)
from models.recommender import generate_recommendations
from reports.llm_reports import (
    generate_student_report, generate_cohort_report, is_api_configured
)


st.set_page_config(
    page_title="UTP Academic Advisor",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --utp-morado: #7a1c79;
        --utp-morado-light: #f3e6f3;
        --utp-dorado: #b98f00;
        --utp-dorado-light: #fdf1d4;
        --utp-dorado-text: #8a6b00;
        --utp-rojo: #e52713;
        --utp-rojo-light: #fbe1de;
        --utp-rojo-text: #c41f0f;
        --utp-verde: #004411;
        --utp-verde-light: #ddeee1;
        --utp-negro: #1d1d1b;
        --utp-card-bg: #ffffff;
        --utp-card-border: #e6dbe6;
        --utp-text-muted: #5a5a5a;
    }

    [data-testid="stSidebar"] { background-color: var(--utp-morado) !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .sidebar-logo { text-align: center; padding: 1rem 0; }

    .kpi-card {
        background: var(--utp-card-bg);
        border: 1px solid var(--utp-card-border);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 5px solid var(--utp-morado);
        box-shadow: 0 2px 6px rgba(29,29,27,0.08);
        color: var(--utp-negro);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: var(--utp-morado); }
    .kpi-label { font-size: 0.85rem; color: var(--utp-text-muted); margin-top: 0.3rem; }

    .riesgo-alto    { background: var(--utp-dorado-light); border-left-color: var(--utp-dorado); }
    .riesgo-critico { background: var(--utp-rojo-light); border-left-color: var(--utp-rojo); }

    .section-title {
        font-size: 1.4rem; font-weight: 700;
        color: var(--utp-morado); border-bottom: 2px solid var(--utp-morado);
        padding-bottom: 0.4rem; margin-bottom: 1rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green  { background: var(--utp-verde-light); color: var(--utp-verde); }
    .badge-blue   { background: var(--utp-morado-light); color: var(--utp-morado); }
    .badge-yellow { background: var(--utp-dorado-light); color: var(--utp-dorado-text); }
    .badge-red    { background: var(--utp-rojo-light); color: var(--utp-rojo-text); }

    .report-box {
        background: var(--utp-card-bg);
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid var(--utp-card-border);
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
        color: var(--utp-negro);
    }

    /* Tarjeta generica reutilizable: fondo claro + texto oscuro garantizado,
       evita el problema de texto invisible (blanco sobre blanco / oscuro sobre oscuro) */
    .info-card {
        background: var(--utp-card-bg);
        border: 1px solid var(--utp-card-border);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: var(--utp-negro);
    }
    .info-card, .info-card * { color: var(--utp-negro); }

    div[data-testid="stSelectbox"] label { font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; }
    .stTabs [aria-selected="true"] { color: var(--utp-morado) !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_process_data():
    """
    Carga, limpia y procesa todos los datos del pipeline.
    El decorador cache_data evita recargar en cada interaccion del usuario.
    """
    raw = load_all_data()
    processed = run_preprocessing(raw)
    df = create_features(processed["merged"])
    return df, processed["courses"], processed["grades"]


@st.cache_data(show_spinner=False)
def run_clustering_cached(df_hash: str, _df: pd.DataFrame):
    """
    Entrena el modelo de clustering y lo guarda en cache.
    El parametro df_hash garantiza que el cache se invalide si cambian los datos.
    """
    return train_clustering(_df)


def get_cluster_badge(label: str) -> str:
    """Devuelve el HTML del badge de color segun el perfil del cluster."""
    colores = {
        "Alto Rendimiento":    "badge-green",
        "Rendimiento Regular": "badge-blue",
        "En Riesgo":           "badge-yellow",
        "Riesgo Critico":      "badge-red"
    }
    clase = colores.get(label, "badge-blue")
    return f'<span class="badge {clase}">{label}</span>'


def sidebar_navigation() -> str:
    """Renderiza la barra lateral con el menu de navegacion."""
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <h2 style="color:#ffffff; font-size:1.3rem; margin:0;">UTP Academic Advisor</h2>
        <p style="color:#e8c97a; font-size:0.8rem; margin-top:0.3rem;">
            Sistema de Recomendacion Academica
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    paginas = {
        "Panel Principal":         "panel",
        "Perfiles de Estudiantes": "perfiles",
        "Estudiantes en Riesgo":   "riesgo",
        "Recomendaciones":         "recomendaciones",
        "Informes con IA":         "informes"
    }

    pagina_actual = st.sidebar.radio(
        "Navegacion",
        list(paginas.keys()),
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <p style="font-size:0.75rem; color:#e8c97a; text-align:center;">
        Universidad Tecnologica de Panama<br>
        Pipeline de Datos Academicos v1.0
    </p>
    """, unsafe_allow_html=True)

    return paginas[pagina_actual]


def page_panel_principal(df: pd.DataFrame, clustering_result: dict):
    """Pagina 1: Panel general con KPIs y graficas de distribucion."""

    st.markdown('<p class="section-title">Panel General de Rendimiento Academico</p>',
                unsafe_allow_html=True)

    total = len(df)
    en_riesgo = len(df[df["estado_academico"].isin(["En Riesgo", "Suspension Academica"])])
    gpa_prom = df["promedio_general"].mean()
    asistencia_prom = df["asistencia_promedio"].mean()
    tasa_rep_prom = df["tasa_reprobacion"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total}</div>
            <div class="kpi-label">Total Estudiantes</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card riesgo-critico">
            <div class="kpi-value" style="color:#c41f0f;">{en_riesgo}</div>
            <div class="kpi-label">En Riesgo o Suspension</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{gpa_prom:.1f}</div>
            <div class="kpi-label">GPA Promedio General</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{asistencia_prom:.1f}%</div>
            <div class="kpi-label">Asistencia Promedio</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        est_criticos = len(df[df["cluster_label"] == "Riesgo Critico"])
        st.markdown(f"""
        <div class="kpi-card riesgo-alto">
            <div class="kpi-value" style="color:#8a6b00;">{est_criticos}</div>
            <div class="kpi-label">Riesgo Critico</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Distribucion de Estudiantes por Facultad**")
        dist_fac = df.groupby("facultad_codigo").size().reset_index(name="count")
        dist_fac["facultad_codigo"] = dist_fac["facultad_codigo"].astype(str)
        fig_fac = px.bar(
            dist_fac,
            x="count",
            y="facultad_codigo",
            orientation="h",
            color="count",
            color_continuous_scale=["#f3e6f3", "#7a1c79"],
            labels={"count": "Estudiantes", "facultad_codigo": "Facultad"},
            text="count"
        )
        fig_fac.update_traces(textposition="outside")
        fig_fac.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=20, t=10, b=0),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_fac, use_container_width=True)

    with col_b:
        st.markdown("**Distribucion por Perfil de Cluster**")
        colores_cluster = {k: v["color"] for k, v in CLUSTER_CONFIG.items()}
        dist_cluster = df["cluster_label"].value_counts().reset_index()
        dist_cluster.columns = ["perfil", "count"]
        dist_cluster["color"] = dist_cluster["perfil"].map(colores_cluster)
        fig_pie = px.pie(
            dist_cluster,
            names="perfil",
            values="count",
            color="perfil",
            color_discrete_map=colores_cluster,
            hole=0.45
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**GPA Promedio por Semestre**")
        gpa_sem = df.groupby("semestre_actual")["promedio_general"].mean().reset_index()
        gpa_sem.columns = ["semestre", "gpa_promedio"]
        fig_sem = px.line(
            gpa_sem,
            x="semestre",
            y="gpa_promedio",
            markers=True,
            labels={"semestre": "Semestre Actual", "gpa_promedio": "GPA Promedio"},
            color_discrete_sequence=["#7a1c79"]
        )
        fig_sem.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        fig_sem.update_xaxes(tickmode="linear", dtick=1)
        st.plotly_chart(fig_sem, use_container_width=True)

    with col_d:
        st.markdown("**Score de Riesgo Promedio por Facultad y Perfil**")
        riesgo_fac = df.groupby(["facultad_codigo", "cluster_label"])["riesgo_score"].mean().reset_index()
        colores_cluster = {k: v["color"] for k, v in CLUSTER_CONFIG.items()}
        fig_riesgo = px.bar(
            riesgo_fac,
            x="facultad_codigo",
            y="riesgo_score",
            color="cluster_label",
            color_discrete_map=colores_cluster,
            barmode="group",
            labels={
                "facultad_codigo": "Facultad",
                "riesgo_score": "Score de Riesgo",
                "cluster_label": "Perfil"
            }
        )
        fig_riesgo.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_riesgo, use_container_width=True)

    with st.expander("Ver tabla resumen por cluster", expanded=False):
        stats = get_cluster_statistics(df)
        stats_display = stats.copy()
        stats_display["tasa_reprobacion"] = (stats_display["tasa_reprobacion"] * 100).round(1).astype(str) + "%"
        stats_display.columns = [
            "Perfil", "N Estudiantes", "GPA Promedio", "Asistencia %",
            "Tasa Reprobacion", "Avance Carrera %", "Score Riesgo"
        ]
        st.dataframe(stats_display, use_container_width=True, hide_index=True)

    st.caption(
        f"Silhouette score del modelo de clustering: "
        f"{clustering_result['silhouette']:.4f} "
        f"(rango -1 a 1, valores mayores a 0.3 indican clusters bien definidos)"
    )


def page_perfiles_estudiantes(df: pd.DataFrame):
    """Pagina 2: Visualizacion interactiva de los perfiles de clustering."""

    st.markdown('<p class="section-title">Perfiles de Estudiantes por Clustering</p>',
                unsafe_allow_html=True)

    col_filtros, col_info = st.columns([2, 1])
    with col_filtros:
        facultades_disponibles = ["Todas"] + sorted(df["facultad_codigo"].unique().tolist())
        fac_sel = st.selectbox("Filtrar por facultad", facultades_disponibles)

    with col_info:
        semestre_rango = st.slider(
            "Rango de semestre",
            min_value=1,
            max_value=10,
            value=(1, 10)
        )

    df_filtrado = df.copy()
    if fac_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado["facultad_codigo"] == fac_sel]
    df_filtrado = df_filtrado[
        (df_filtrado["semestre_actual"] >= semestre_rango[0]) &
        (df_filtrado["semestre_actual"] <= semestre_rango[1])
    ]

    colores_cluster = {k: v["color"] for k, v in CLUSTER_CONFIG.items()}

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**GPA vs Asistencia por Perfil**")
        fig_scatter = px.scatter(
            df_filtrado,
            x="asistencia_promedio",
            y="promedio_general",
            color="cluster_label",
            color_discrete_map=colores_cluster,
            hover_data=["nombre_completo", "carrera", "semestre_actual", "riesgo_score"],
            labels={
                "asistencia_promedio": "Asistencia Promedio (%)",
                "promedio_general": "Promedio General (GPA)",
                "cluster_label": "Perfil"
            },
            opacity=0.75,
            size_max=8
        )
        fig_scatter.add_hline(y=70, line_dash="dot", line_color="gray",
                              annotation_text="GPA minimo (70)")
        fig_scatter.add_vline(x=70, line_dash="dot", line_color="gray",
                              annotation_text="Asistencia minima")
        fig_scatter.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_g2:
        st.markdown("**Tasa de Reprobacion vs Avance en Carrera**")
        fig_rep = px.scatter(
            df_filtrado,
            x="avance_carrera",
            y="tasa_reprobacion",
            color="cluster_label",
            color_discrete_map=colores_cluster,
            hover_data=["nombre_completo", "facultad_codigo", "promedio_general"],
            labels={
                "avance_carrera": "Avance en Carrera (%)",
                "tasa_reprobacion": "Tasa de Reprobacion",
                "cluster_label": "Perfil"
            },
            opacity=0.75
        )
        fig_rep.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01)
        )
        st.plotly_chart(fig_rep, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Descripcion de cada Perfil Academico**")
    cols_perfil = st.columns(4)
    for i, (perfil, config) in enumerate(CLUSTER_CONFIG.items()):
        with cols_perfil[i]:
            n_est = len(df_filtrado[df_filtrado["cluster_label"] == perfil])
            prom = df_filtrado[df_filtrado["cluster_label"] == perfil]["promedio_general"].mean()
            prom_str = f"{prom:.1f}" if not pd.isna(prom) else "N/A"
            color = config['color']
            descripcion = config['descripcion']
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; border: 1px solid #e6dbe6;
                        padding: 0.8rem; border-radius: 0 8px 8px 0;
                        background: #ffffff; color:#1d1d1b; height: 100%;">
                <strong style="color:{color};">{perfil}</strong><br>
                <small style="color:#5a5a5a;">{descripcion}</small><br><br>
                <b>{n_est}</b> estudiantes | GPA prom: <b>{prom_str}</b>
            </div>
            """, unsafe_allow_html=True)


def page_estudiantes_en_riesgo(df: pd.DataFrame, grades: pd.DataFrame, courses: pd.DataFrame):
    """Pagina 3: Identificacion y detalle de estudiantes en situacion de riesgo."""

    st.markdown('<p class="section-title">Identificacion de Estudiantes en Riesgo</p>',
                unsafe_allow_html=True)

    df_riesgo = df[df["cluster_label"].isin(["En Riesgo", "Riesgo Critico"])].copy()
    df_riesgo = df_riesgo.sort_values("riesgo_score", ascending=False)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        fac_options = ["Todas"] + sorted(df_riesgo["facultad_codigo"].unique().tolist())
        fac_filter = st.selectbox("Filtrar por facultad", fac_options, key="fac_riesgo")
    with col_f2:
        perfil_options = ["Ambos", "En Riesgo", "Riesgo Critico"]
        perfil_filter = st.selectbox("Nivel de riesgo", perfil_options)
    with col_f3:
        sem_filter = st.selectbox(
            "Semestre",
            ["Todos"] + [str(i) for i in range(1, 11)],
            key="sem_riesgo"
        )

    if fac_filter != "Todas":
        df_riesgo = df_riesgo[df_riesgo["facultad_codigo"] == fac_filter]
    if perfil_filter != "Ambos":
        df_riesgo = df_riesgo[df_riesgo["cluster_label"] == perfil_filter]
    if sem_filter != "Todos":
        df_riesgo = df_riesgo[df_riesgo["semestre_actual"] == int(sem_filter)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Estudiantes en pantalla", len(df_riesgo))
    c2.metric("GPA Promedio del Grupo", f"{df_riesgo['promedio_general'].mean():.1f}" if len(df_riesgo) > 0 else "N/A")
    c3.metric("Score de Riesgo Promedio", f"{df_riesgo['riesgo_score'].mean():.1f}" if len(df_riesgo) > 0 else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    if df_riesgo.empty:
        st.info("No se encontraron estudiantes con los filtros aplicados.")
        return

    for _, row in df_riesgo.head(20).iterrows():
        badge_html = get_cluster_badge(row["cluster_label"])
        col_header, col_badge = st.columns([4, 1])
        with col_header:
            with st.expander(
                f"{row['nombre_completo']} | {row['facultad_codigo']} "
                f"| Sem {row['semestre_actual']} | GPA: {row['promedio_general']:.1f}",
                expanded=False
            ):
                c_a, c_b, c_c = st.columns(3)
                with c_a:
                    st.markdown(f"**Carrera:** {row['carrera']}")
                    st.markdown(f"**Semestre:** {row['semestre_actual']}")
                    st.markdown(f"**Estado:** {row['estado_academico']}")
                    st.markdown(f"**Correo:** {row['correo']}")
                with c_b:
                    st.markdown(f"**GPA:** {row['promedio_general']:.1f} / 100")
                    st.markdown(f"**Asistencia:** {row['asistencia_promedio']:.1f}%")
                    st.markdown(f"**Tasa de reprobacion:** {row['tasa_reprobacion']*100:.1f}%")
                    st.markdown(f"**Avance en carrera:** {row.get('avance_carrera', 0):.1f}%")
                with c_c:
                    creditos_ap = row.get("creditos_aprobados", 0)
                    creditos_rep = row.get("creditos_reprobados", 0)
                    st.markdown(f"**Creditos aprobados:** {creditos_ap}")
                    st.markdown(f"**Creditos reprobados:** {creditos_rep}")
                    st.markdown(f"**Score de riesgo:** {row['riesgo_score']:.1f} / 100")
                    st.markdown(f"**Perfil:** ", unsafe_allow_html=False)
                    st.markdown(badge_html, unsafe_allow_html=True)

                st.markdown("**Recomendacion del sistema:**")
                config = CLUSTER_CONFIG.get(row["cluster_label"], {})
                st.info(config.get("recomendacion_general", "Sin recomendacion disponible."))

    if len(df_riesgo) > 20:
        st.caption(f"Mostrando 20 de {len(df_riesgo)} estudiantes en riesgo.")


def page_recomendaciones(df: pd.DataFrame, courses: pd.DataFrame, grades: pd.DataFrame):
    """Pagina 4: Sistema de recomendacion de materias por estudiante."""

    st.markdown('<p class="section-title">Sistema de Recomendacion de Materias</p>',
                unsafe_allow_html=True)

    col_sel, col_info = st.columns([2, 2])

    with col_sel:
        opciones_estudiantes = df["student_id"].tolist()
        nombres_dict = dict(zip(df["student_id"], df["nombre_completo"]))
        opciones_display = [f"{sid} - {nombres_dict[sid]}" for sid in opciones_estudiantes]

        seleccion = st.selectbox(
            "Seleccionar estudiante",
            opciones_display,
            key="sel_estudiante_rec"
        )
        student_id_sel = seleccion.split(" - ")[0]

    student_row = df[df["student_id"] == student_id_sel].iloc[0]

    with col_info:
        st.markdown(f"""
        <div class="info-card" style="margin-top:1.6rem;">
            <b>{student_row['nombre_completo']}</b><br>
            {student_row['carrera']} | {student_row['facultad_codigo']}<br>
            Semestre {student_row['semestre_actual']} |
            GPA {student_row['promedio_general']:.1f} |
            {get_cluster_badge(student_row['cluster_label'])}
        </div>
        """, unsafe_allow_html=True)

    n_recomendaciones = st.slider("Numero de recomendaciones", min_value=3, max_value=10, value=5)

    with st.spinner("Calculando recomendaciones..."):
        recomendaciones = generate_recommendations(
            student_id_sel, df, courses, grades, top_n=n_recomendaciones
        )

    if recomendaciones.empty:
        st.warning("No se encontraron materias elegibles para este estudiante. Puede que haya completado todos los cursos disponibles.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Materias Recomendadas**")

    colores_dificultad = {1: "#004411", 2: "#4a7a3d", 3: "#b98f00", 4: "#d1571b", 5: "#e52713"}

    for _, rec in recomendaciones.iterrows():
        col_num, col_main, col_meta = st.columns([0.5, 5, 2])
        with col_num:
            st.markdown(f"""
            <div style="background:#7a1c79; color:white; border-radius:50%;
                        width:36px; height:36px; display:flex;
                        align-items:center; justify-content:center;
                        font-weight:700; font-size:1rem; margin-top:0.3rem;">
                {_}
            </div>""", unsafe_allow_html=True)

        with col_main:
            color_dif = colores_dificultad.get(int(rec["nivel_dificultad"]), "#999")
            st.markdown(f"""
            <div style="border:1px solid #e6dbe6; border-radius:8px;
                        padding:0.8rem 1rem; background:#ffffff; color:#1d1d1b;">
                <b style="font-size:1rem;">{rec['nombre']}</b>
                <span style="color:#5a5a5a; font-size:0.85rem;"> ({rec['codigo']})</span><br>
                <span style="color:#3d3d3b; font-size:0.85rem;">{rec['descripcion']}</span><br>
                <span style="font-size:0.8rem; color:#5a5a5a; margin-top:0.3rem; display:block;">
                    {rec['razon']}
                </span>
            </div>""", unsafe_allow_html=True)

        with col_meta:
            puntaje_pct = int(rec["puntaje_recomendacion"] * 100)
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e6dbe6; border-radius:8px;
                        text-align:center; padding:0.6rem; color:#1d1d1b;">
                <div style="font-size:1.5rem; font-weight:700; color:#7a1c79;">{puntaje_pct}%</div>
                <div style="font-size:0.75rem; color:#5a5a5a;">Compatibilidad</div>
                <br>
                <div>
                    <span style="color:{color_dif}; font-weight:600;">
                        {'|' * int(rec['nivel_dificultad'])} Dif. {int(rec['nivel_dificultad'])}/5
                    </span>
                </div>
                <div style="font-size:0.8rem; color:#5a5a5a;">
                    {int(rec['creditos'])} creditos | Sem. {int(rec['semestre_recomendado'])}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Ver tabla completa de recomendaciones"):
        tabla = recomendaciones[["codigo", "nombre", "area_conocimiento",
                                  "creditos", "nivel_dificultad",
                                  "semestre_recomendado", "puntaje_recomendacion"]].copy()
        tabla["puntaje_recomendacion"] = (tabla["puntaje_recomendacion"] * 100).round(1).astype(str) + "%"
        tabla.columns = ["Codigo", "Nombre", "Area", "Creditos",
                         "Dificultad", "Semestre Rec.", "Compatibilidad"]
        st.dataframe(tabla, use_container_width=True, hide_index=True)


def page_informes_ia(df: pd.DataFrame, courses: pd.DataFrame, grades: pd.DataFrame):
    """Pagina 5: Generacion de informes academicos personalizados con LLM."""

    st.markdown('<p class="section-title">Generacion de Informes Academicos con IA</p>',
                unsafe_allow_html=True)

    if not is_api_configured():
        st.warning(
            "La API de Groq no esta configurada. "
            "Crea una cuenta gratuita en https://console.groq.com, "
            "obtén tu API key y agregala al archivo .env como GROQ_API_KEY=tu_clave"
        )

    tab_individual, tab_cohorte = st.tabs([
        "Informe Individual",
        "Informe de Cohorte"
    ])

    with tab_individual:
        st.markdown("Selecciona un estudiante para generar un informe academico personalizado.")

        nombres_dict = dict(zip(df["student_id"], df["nombre_completo"]))
        opciones_display = [f"{sid} - {nombres_dict[sid]}" for sid in df["student_id"]]
        seleccion = st.selectbox("Estudiante", opciones_display, key="sel_informe")
        student_id_sel = seleccion.split(" - ")[0]

        student_row = df[df["student_id"] == student_id_sel].iloc[0]

        col_preview, col_btn = st.columns([3, 1])
        with col_preview:
            st.markdown(f"""
            <div class="info-card">
                <b>{student_row['nombre_completo']}</b> |
                {student_row['carrera']} |
                Sem. {student_row['semestre_actual']} |
                GPA {student_row['promedio_general']:.1f} |
                {get_cluster_badge(student_row['cluster_label'])}
            </div>
            """, unsafe_allow_html=True)

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generar = st.button("Generar Informe", type="primary", key="btn_informe_ind")

        if generar:
            with st.spinner("Generando informe con IA... esto puede tardar unos segundos."):
                recomendaciones = generate_recommendations(
                    student_id_sel, df, courses, grades, top_n=5
                )
                informe = generate_student_report(student_row, recomendaciones)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Informe Generado**")
            st.markdown(f'<div class="report-box">{informe}</div>', unsafe_allow_html=True)
            st.download_button(
                label="Descargar Informe (.txt)",
                data=informe,
                file_name=f"informe_{student_id_sel}.txt",
                mime="text/plain"
            )

    with tab_cohorte:
        st.markdown("Genera un informe ejecutivo del estado academico general de todos los estudiantes.")

        col_fac, col_btn2 = st.columns([3, 1])
        with col_fac:
            fac_ops = ["Toda la UTP"] + sorted(df["facultad_codigo"].unique().tolist())
            fac_cohorte = st.selectbox("Alcance del informe", fac_ops, key="cohorte_fac")
        with col_btn2:
            st.markdown("<br>", unsafe_allow_html=True)
            generar_cohorte = st.button("Generar Informe", type="primary", key="btn_informe_coh")

        if generar_cohorte:
            df_scope = df if fac_cohorte == "Toda la UTP" else df[df["facultad_codigo"] == fac_cohorte]
            n_en_riesgo = len(df_scope[df_scope["cluster_label"].isin(["En Riesgo", "Riesgo Critico"])])

            from models.clustering import get_cluster_statistics
            stats = get_cluster_statistics(df_scope)

            with st.spinner("Generando informe de cohorte con IA..."):
                informe_coh = generate_cohort_report(stats, len(df_scope), n_en_riesgo)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Informe de Cohorte**")
            st.markdown(f'<div class="report-box">{informe_coh}</div>', unsafe_allow_html=True)
            st.download_button(
                label="Descargar Informe de Cohorte (.txt)",
                data=informe_coh,
                file_name=f"informe_cohorte_{fac_cohorte.replace(' ', '_')}.txt",
                mime="text/plain"
            )


def main():
    """Punto de entrada principal de la aplicacion Streamlit."""

    pagina = sidebar_navigation()

    with st.spinner("Cargando datos y ejecutando pipeline..."):
        try:
            df_raw, courses, grades = load_and_process_data()
        except FileNotFoundError as e:
            st.error(str(e))
            st.code("python data/generate_data.py", language="bash")
            st.stop()

    df_hash = str(len(df_raw)) + str(df_raw["promedio_general"].sum())
    clustering_result = run_clustering_cached(df_hash, df_raw)
    df = assign_cluster_labels(df_raw, clustering_result)

    if pagina == "panel":
        page_panel_principal(df, clustering_result)

    elif pagina == "perfiles":
        page_perfiles_estudiantes(df)

    elif pagina == "riesgo":
        page_estudiantes_en_riesgo(df, grades, courses)

    elif pagina == "recomendaciones":
        page_recomendaciones(df, courses, grades)

    elif pagina == "informes":
        page_informes_ia(df, courses, grades)


if __name__ == "__main__":
    main()