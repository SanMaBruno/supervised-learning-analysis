<<<<<<< HEAD
# supervised-learning-analysis
Análisis de aprendizaje supervisado 
=======
 
# Análisis de Datos Supervisados: Cáncer de Mama y Titanic

Este proyecto realiza un análisis de datos utilizando algoritmos de aprendizaje supervisado aplicados a dos datasets: `load_breast_cancer` y el dataset del Titanic. La meta es comparar el rendimiento de diferentes modelos supervisados para clasificación, como Naive Bayes, Análisis Discriminante Lineal (LDA) y Regresión Logística.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:



supervised-learning-analysis/
│
├── data/                        # Directorio para los datos (no se sube a GitHub)
│   ├── wdbc.data                # Dataset de cáncer de mama
│   ├── titanic_data.csv         # Dataset del Titanic
│
├── notebooks/                   # Notebooks Jupyter
│   ├── analysis.ipynb           # Análisis principal y evaluación de modelos
│
├── results/                     # Resultados generados por el análisis
│   ├── figures/                 # Directorio para guardar las gráficas de confusión
│   ├── models/                  # Modelos entrenados guardados como archivos .pkl
│
├── src/                         # Código fuente del proyecto
│   ├── data_preprocessing.py    # Preprocesamiento de los datos
│   ├── feature_engineering.py   # Ingeniería de características
│   ├── models.py                # Definición y entrenamiento de modelos
│   ├── evaluation.py            # Funciones para evaluación de modelos
│
├── .gitignore                   # Archivos y carpetas ignoradas por Git
└── README.md                    # Descripción del proyecto


## Descripción del Análisis

### 1. Dataset de Cáncer de Mama
El primer análisis utiliza el dataset de cáncer de mama (`load_breast_cancer`). Los datos se preprocesan, se dividen en conjuntos de entrenamiento y prueba, y se aplican tres modelos supervisados: 
- Naive Bayes
- Análisis Discriminante Lineal (LDA)
- Regresión Logística

El rendimiento de los modelos se evalúa utilizando la precisión, la matriz de confusión y el informe de clasificación.

### 2. Dataset del Titanic
El segundo análisis utiliza el dataset del Titanic. Similar al primer análisis, se preprocesan los datos, eliminando las columnas irrelevantes. Luego, se entrenan los mismos modelos supervisados y se evalúan utilizando métricas similares.

### Resultados
Las matrices de confusión y los informes de clasificación muestran cómo se desempeñan los modelos en cada dataset. En este proyecto, se incluye un análisis detallado del rendimiento de cada modelo, junto con las conclusiones y posibles mejoras futuras.

## Cómo Ejecutar el Proyecto
1. Clona este repositorio:
    ```bash
    git clone https://github.com/SanMaBruno/supervised-learning-analysis.git
    ```

2. Navega al directorio del proyecto:
    ```bash
    cd supervised-learning-analysis
    ```

3. Crea y activa un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/MacOS
    venv\Scripts\activate  # En Windows
    ```

4. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

5. Ejecuta los notebooks para ver el análisis completo.

## Requisitos
- Python 3.8 o superior
- scikit-learn
- pandas
- numpy
- matplotlib

## Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un "issue" o un "pull request" para mejoras o correcciones.

## Autor
Desarrollado por [Bruno San Martín](https://github.com/SanMaBruno).
>>>>>>> 349cc41 (Initial commit: Add data preprocessing, feature engineering, and model training scripts)
