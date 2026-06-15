# Sistema de Recomendacion Academica UTP

Sistema inteligente de gestion de datos estudiantiles para la Universidad Tecnologica de Panama.
Combina un pipeline de datos de dos fuentes, clustering con KMeans y un sistema de recomendacion hibrido,
todo accesible desde un dashboard interactivo construido con Streamlit.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![LLM](https://img.shields.io/badge/LLM-Groq%20%28gratuito%29-green)
![License](https://img.shields.io/badge/Licencia-MIT-lightgrey)


## Tabla de Contenidos

1. [Descripcion del Proyecto](#descripcion-del-proyecto)
2. [Problema que Resuelve](#problema-que-resuelve)
3. [Arquitectura del Sistema](#arquitectura-del-sistema)
4. [Funcionalidades del Dashboard](#funcionalidades-del-dashboard)
5. [Tecnologias Utilizadas](#tecnologias-utilizadas)
6. [Estructura del Repositorio](#estructura-del-repositorio)
7. [Instalacion y Configuracion](#instalacion-y-configuracion)
8. [Uso](#uso)
9. [Datasets Simulados](#datasets-simulados)
10. [Modelos de Machine Learning](#modelos-de-machine-learning)
11. [API de LLM Gratuita](#api-de-llm-gratuita)
12. [Autores](#autores)


## Descripcion del Proyecto

Este proyecto implementa un sistema de recomendacion academica que ayuda a docentes y asesores de la UTP
a identificar estudiantes en riesgo, sugerir materias apropiadas para cada perfil y generar informes
automatizados con inteligencia artificial.

El sistema procesa datos de rendimiento academico de 300 estudiantes distribuidos en 6 facultades,
aplica clustering no supervisado para segmentar perfiles, y ejecuta un motor de recomendacion hibrido
que considera afinidad por area, nivel de dificultad, exito historico del cluster y avance academico.


## Problema que Resuelve

La UTP atiende miles de estudiantes en diferentes carreras de ingenieria. Los docentes frecuentemente
no cuentan con herramientas que les permitan identificar de forma rapida y objetiva quienes estan en
riesgo de desercion o suspension, ni que materias son mas convenientes para cada estudiante segun
su perfil real. Este sistema centraliza esa informacion y la convierte en decisiones accionables.


## Arquitectura del Sistema

```
Fuente 1: students_performance.csv       Fuente 2: courses_catalog.csv
           (rendimiento estudiantil)                (catalogo UTP)
                    |                                      |
                    +-------------- pipeline/ --------------+
                                         |
                              ingestion.py       <- carga de datos
                              preprocessing.py  <- limpieza y merge
                              feature_engineering.py  <- features ML
                                         |
                                     models/
                              clustering.py    <- KMeans k=4
                              recommender.py   <- scoring hibrido
                                         |
                                     reports/
                              llm_reports.py   <- Groq LLM
                                         |
                                      app.py   <- Dashboard Streamlit
```


## Funcionalidades del Dashboard

**Panel Principal**
Vista general con KPIs del sistema (total estudiantes, cantidad en riesgo, GPA promedio, asistencia
promedio) junto a graficas de distribucion por facultad y por perfil de cluster.

**Perfiles de Estudiantes**
Visualizacion interactiva del resultado del clustering. Scatter plots de GPA vs asistencia y tasa de
reprobacion vs avance en carrera, filtrados por facultad y semestre.

**Estudiantes en Riesgo**
Lista detallada de estudiantes en los perfiles "En Riesgo" y "Riesgo Critico" con filtros por facultad,
nivel de riesgo y semestre. Cada estudiante tiene un expandible con sus indicadores completos.

**Recomendaciones**
Motor de recomendacion hibrido que genera las top N materias para cualquier estudiante seleccionado.
Muestra el puntaje de compatibilidad, la razon de la recomendacion y los metadatos del curso.

**Informes con IA**
Genera informes academicos personalizados (por estudiante) o ejecutivos (por cohorte o facultad)
usando el LLM Llama 3.1 via Groq API. Los informes son descargables en formato .txt.


## Tecnologias Utilizadas

| Categoria          | Tecnologia                        |
|--------------------|-----------------------------------|
| Lenguaje           | Python 3.10+                      |
| Dashboard          | Streamlit 1.28+                   |
| Machine Learning   | scikit-learn (KMeans, Silhouette) |
| Visualizacion      | Plotly Express / Graph Objects    |
| Procesamiento      | Pandas, NumPy                     |
| LLM (gratuito)     | Groq API con Llama 3.1 8B         |
| Variables entorno  | python-dotenv                     |


## Estructura del Repositorio

```
utp-academic-system/
|
+-- app.py                          Dashboard principal de Streamlit
+-- requirements.txt                Dependencias del proyecto
+-- .env.example                    Plantilla de variables de entorno
+-- .gitignore
+-- README.md
|
+-- data/
|   +-- generate_data.py            Generador de datasets simulados UTP
|   +-- students_performance.csv    Fuente 1: rendimiento estudiantil (generado)
|   +-- courses_catalog.csv         Fuente 2: catalogo de materias (generado)
|   +-- student_grades.csv          Fuente 3: historial de notas (generado)
|
+-- pipeline/
|   +-- __init__.py
|   +-- ingestion.py                Carga de las fuentes de datos
|   +-- preprocessing.py           Limpieza, normalizacion y merge
|   +-- feature_engineering.py     Creacion de features para ML
|
+-- models/
|   +-- __init__.py
|   +-- clustering.py              KMeans con asignacion de etiquetas semanticas
|   +-- recommender.py             Motor de recomendacion hibrido (4 componentes)
|
+-- reports/
    +-- __init__.py
    +-- llm_reports.py             Generacion de informes con Groq LLM
```


## Instalacion y Configuracion

### Requisitos previos

- Python 3.10 o superior
- pip actualizado

### Pasos de instalacion

1. Clona el repositorio

```bash
git clone https://github.com/tu_usuario/utp-academic-system.git
cd utp-academic-system
```

2. Crea y activa un entorno virtual

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

3. Instala las dependencias

```bash
pip install -r requirements.txt
```

4. Configura la API key de Groq (opcional pero recomendado)

```bash
cp .env.example .env
# Edita el archivo .env con tu API key gratuita de https://console.groq.com
```

5. Genera los datasets simulados

```bash
python data/generate_data.py
```

6. Inicia el dashboard

```bash
streamlit run app.py
```

El dashboard estara disponible en http://localhost:8501


## Uso

Luego de iniciar la aplicacion, navega entre las paginas usando el menu lateral:

1. Comienza en el **Panel Principal** para ver el estado general de la institucion.
2. Explora los **Perfiles de Estudiantes** para entender la segmentacion por cluster.
3. Revisa los **Estudiantes en Riesgo** para identificar casos de atencion prioritaria.
4. Selecciona un estudiante en **Recomendaciones** para ver sus materias sugeridas.
5. En **Informes con IA**, genera un informe academico con un solo clic.


## Datasets Simulados

El sistema genera tres archivos CSV con datos realistas de la UTP:

**students_performance.csv (Fuente 1)**
300 estudiantes distribuidos en 6 facultades (FISC, FIC, FIE, FIEM, FIIA, FACIN).
Columnas principales: student_id, nombre_completo, facultad_codigo, carrera, semestre_actual,
promedio_general, asistencia_promedio, creditos_aprobados, tasa_reprobacion, estado_academico.

**courses_catalog.csv (Fuente 2)**
80 materias entre comunes (primer y segundo ano) y especificas por facultad.
Columnas principales: course_id, codigo, nombre, facultad, semestre_recomendado, creditos,
area_conocimiento, nivel_dificultad, prerequisitos_ids.

**student_grades.csv (Fuente 3)**
Historial de notas por estudiante y materia, con nota, estado de aprobacion y numero de intentos.


## Modelos de Machine Learning

### Clustering con KMeans (k=4)

Features utilizadas:
- promedio_general: GPA del estudiante (0-100)
- asistencia_promedio: porcentaje de asistencia
- tasa_reprobacion: proporcion de materias reprobadas
- avance_carrera: creditos aprobados sobre creditos esperados por semestre
- intensidad_academica: materias cursadas por semestre

Las features se normalizan con StandardScaler antes de aplicar KMeans.
Los clusters se etiquetan semanticamente ordenando los centroides por promedio_general:

| Cluster           | GPA Tipico | Asistencia Tipica | Descripcion                          |
|-------------------|-----------|-------------------|--------------------------------------|
| Alto Rendimiento  | 85 - 100  | 90 - 100%         | Perfil excelente, candidatos avance  |
| Rendimiento Reg.  | 70 - 84   | 75 - 90%          | Perfil solido, requiere mantenimiento|
| En Riesgo         | 55 - 69   | 60 - 75%          | Alertas activas, atencion recomendada|
| Riesgo Critico    | 0 - 54    | 0 - 60%           | Intervencion inmediata requerida     |

### Motor de Recomendacion Hibrido

El puntaje de cada materia candidata se calcula con cuatro componentes:

```
puntaje = 0.30 * afinidad_por_area
        + 0.25 * compatibilidad_dificultad
        + 0.25 * tasa_exito_en_cluster
        + 0.20 * compatibilidad_de_semestre
```

Adicionalmente se aplican filtros de elegibilidad: la materia debe pertenecer a la facultad del
estudiante o ser comun, y todos sus prerequisitos deben estar aprobados.


## API de LLM Gratuita

Este proyecto usa la API de **Groq** que ofrece inferencia gratuita con el modelo Llama 3.1 8B.

Pasos para obtener tu API key gratuita:

1. Ve a https://console.groq.com y crea una cuenta gratuita.
2. En el panel de la cuenta, haz clic en "API Keys".
3. Crea una nueva clave y copiala.
4. Pega la clave en el archivo .env del proyecto: `GROQ_API_KEY=tu_clave`
5. Reinicia la aplicacion.

Los limites del plan gratuito de Groq son suficientes para uso academico y demostraciones.


## Autores

Desarrollado como proyecto integrador del curso de Ciencia de Datos.
Universidad Tecnologica de Panama.

Si tienes preguntas o sugerencias, abre un Issue en el repositorio.


## Licencia

MIT License. Libre para uso academico y educativo.
