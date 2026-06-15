import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """
Eres un asesor academico experto de la Universidad Tecnologica de Panama (UTP).
Genera informes academicos claros, profesionales y empaticos para docentes.
Reglas: tono profesional, secciones con titulos en mayuscula, sin markdown con asteriscos,
entre 300 y 450 palabras, escribe en español de Panama.
""".strip()


def is_api_configured() -> bool:
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())


def build_student_context(student_row, recommendations):
    nombre    = student_row.get("nombre_completo", "Estudiante")
    facultad  = student_row.get("facultad_nombre", "N/A")
    carrera   = student_row.get("carrera", "N/A")
    semestre  = student_row.get("semestre_actual", "N/A")
    gpa       = student_row.get("promedio_general", 0)
    asistencia= student_row.get("asistencia_promedio", 0)
    tasa_rep  = student_row.get("tasa_reprobacion", 0)
    avance    = student_row.get("avance_carrera", 0)
    estado    = student_row.get("estado_academico", "N/A")
    cluster   = student_row.get("cluster_label", "N/A")
    riesgo    = student_row.get("riesgo_score", 0)

    lineas = []
    if recommendations is not None and not recommendations.empty:
        for _, rec in recommendations.iterrows():
            lineas.append(f"  - {rec['nombre']} ({rec['codigo']}) | Razon: {rec['razon']}")
    recs_texto = "\n".join(lineas) if lineas else "No disponibles."

    return f"""
DATOS DEL ESTUDIANTE
Nombre           : {nombre}
Facultad         : {facultad}
Carrera          : {carrera}
Semestre actual  : {semestre}
Estado academico : {estado}
Perfil cluster   : {cluster}
Score de riesgo  : {riesgo}/100

INDICADORES ACADEMICOS
Promedio general    : {gpa:.1f} / 100
Asistencia promedio : {asistencia:.1f}%
Tasa de reprobacion : {tasa_rep*100:.1f}%
Avance en carrera   : {avance:.1f}%

MATERIAS RECOMENDADAS
{recs_texto}

INSTRUCCION
Genera un informe academico completo dirigido al docente asesor. Incluye: resumen del perfil,
evaluacion del riesgo academico, fortalezas, areas de mejora y plan de accion con las materias recomendadas.
""".strip()


def generate_student_report(student_row, recommendations):
    if not is_api_configured():
        return (
            "INFORME NO DISPONIBLE\n\n"
            "Configura tu GROQ_API_KEY en el archivo .env\n"
            "Obten tu clave gratuita en: https://console.groq.com"
        )
    try:
        contexto = build_student_context(student_row, recommendations)
        payload  = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": contexto}
            ],
            "temperature": 0.6,
            "max_tokens": 700
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        }
        r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR AL GENERAR EL INFORME: {str(e)}"


def generate_cohort_report(stats_df, n_students, n_en_riesgo):
    if not is_api_configured():
        return "Configura tu GROQ_API_KEY en el archivo .env para activar esta funcionalidad."
    try:
        lineas = []
        for _, row in stats_df.iterrows():
            lineas.append(
                f"  Perfil '{row['cluster_label']}': {row['n_estudiantes']} estudiantes | "
                f"GPA {row['promedio_general']:.1f} | "
                f"Asistencia {row['asistencia_promedio']:.1f}% | "
                f"Reprobacion {row['tasa_reprobacion']*100:.1f}%"
            )
        contexto = (
            f"RESUMEN DE COHORTE\n"
            f"Total estudiantes  : {n_students}\n"
            f"Estudiantes riesgo : {n_en_riesgo} ({n_en_riesgo/max(n_students,1)*100:.1f}%)\n\n"
            f"DISTRIBUCION POR PERFILES\n" + "\n".join(lineas) +
            "\n\nINSTRUCCION\nGenera un informe ejecutivo del estado academico general. "
            "Incluye evaluacion global, grupos prioritarios y recomendaciones estrategicas para el cuerpo docente."
        )
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": contexto}
            ],
            "temperature": 0.5,
            "max_tokens": 600
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json"
        }
        r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR AL GENERAR EL INFORME DE COHORTE: {str(e)}"