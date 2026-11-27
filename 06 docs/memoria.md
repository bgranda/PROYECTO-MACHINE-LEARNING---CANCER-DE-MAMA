# 1) INTRODUCCIÓN

## CONCEPTOS BÁSICOS SOBRE EL CÁNCER DE MAMA

El cáncer de mama es un tipo de cáncer que se origina en el tejido mamario debido al crecimiento anormal de células. El cáncer de mama puede darse tanto en hombres como en mujeres.

Clasificaciones para describir el estado de la enfermedad de cáncer de mama:

1) El TNM, que es el sistema estándar internacional para describir la exténsión anatómica del cáncer. En cáncer de mama, las letras significan:

- T (Tumor): tamaño y extensión del tumor primario --> Columna: 'T Stage' y 'Tumor Size'
- N (Nodes): afectación de ganglios linfáticos regionales --> Columna: 'N Stage'
- M (Metastasis): presencia o ausencia de metástasis a distancia. 

2) 6th Stage, que es el estadio clínico del cáncer de mama según la clasificación AJCC (6.ª edición), que refleja el grado de extensión de la enfermedad.

3) Grado de diferenciación tumoral, que explica cuanto se parecen las células cancerígenas a las células mamarias normales --> Columnas: 'Grade' y 'differentiate'.

4) A Stage, que en este dataset define la localización del tumor, es decir, cuánto se ha extendido el tumor. 

5) Estrogen Status y Progesterone Status, indican si el cácer de mama está relacionado hormonalmente o no. Se utiliza para poner el tratamiento al paciente.

6) Regional Node Examined and Regional Node Positive, indican cuántos gánglios linfáticos regionales fueron examinados y cuántos de ellos dieron positivo.


Relaciones entre variables:

Relación TNM y Tumor Size:

| T Stage   | Tumor Size    | N Stage   | M | 
| ---   | ---   | ---   | ---   | 
| T1    | <= 2 cm   | N0    | M0    | 
| T1    | <= 2 cm   | N1    | M0    |
| T2    | > 2 <= 5 cm   | N0    | M0    | 
| T2    | > 2 <= 5 cm   | N1    | M0    | 
| T3    | > 5 cm    | N0 - N1   | M0    |
| T1 - T3   | Cualquier | N2 - N3   | M0    |
| T4    | Invasión de piel o pared torácica | Cualquier | M0    | 
| Cualquiera    | Cualquier | Cualquier | M1    | 

El mejor pronóstico de la enfermedad es cuanto menor sean todos los parámetros anteriores y M = 0 (ausencia de metástasis).

Relación Grade y Differenciate:

| Grade | Differenciate |
| ---   | --- |
| 1 | Well differentiated   |
| 2 | Moderately differentiated |
| 3 | Poorly differentiated |
| 4 | Undifferentiated  |

El mejor pronóstico de la enfermedad es cuanto más diferenciado (grado menor) esté el tumor.

Relación entre A Stage, N Stage y M:

| A Stage   | N Stage   | M |
| ---   | ---   | ---   |
| Localized | N0    | M0    |
| Regional  | N1 - N3   | M0    |
| Distant   | Cualquier | M1    |

El mejor pronóstico es cuanto más localizado esté el tumor.

# 2) FUENTES DE DATOS

El dataset, 'Breast_Cancer.csv' proviente de Kaggle del siguiente autor: Reihaneh Namdari, quien cogió los datos de la web: SEER Program of the NCI.

El autor menciona que son datos de pacientes de cáncer de mama con carcinoma ductal infiltrante y carcinoma lobulillar infiltrante diagnósticadas entre 2006 y 2010.

*Nota del dataset del autor:*

*This dataset of breast cancer patients was obtained from the 2017 November update of the SEER Program of the NCI, which provides information on population-based cancer statistics. The dataset involved female patients with infiltrating duct and lobular carcinoma breast cancer (SEER primary cites recode NOS histology codes 8522/3) diagnosed in 2006-2010. Patients with unknown tumour size, examined regional LNs, positive regional LNs, and patients whose survival months were less than 1 month were excluded; thus, 4024 patients were ultimately included.*

*Autor: Reihaneh Namdari*

# 3) LIMPIEZA DE DATOS

Tenemos un dataset con un tamaño de 4024 x 16, que no tiene valores perdidos.

Los datos parecen que están limpios ya que no tienen valores perdidos.

Son datos clínicos son anonimizados, por lo que se pueden tener valores repetidos y pertenecer a pacientes distintos, por lo que no se podrían eliminar los datos repetidos.

# 4) EL PROBLEMA

Queremos predecir la supervivencia en meses de las pacientes con cancer de mama con carcinoma ductal infiltrante y lobulillar infiltrante a partir del dataset anterior. Este problema es un problema de Regresión Lineal de Machine Learning. Tenemos que estudiar la correlación de todas las variables categóricas y numéricas con la supervivencia en meses, y también si tenemos variables que estén muy relacionadas entre ellas y nos den la misma información.

# 5) DIVIDIR EN TRAIN Y TEST

Como ya hemos dicho en el apartado anterior, nuestro target es 'Survival Months'.

Ahora vamos a dividir nuestros datos en Train y Test, para que todos los aprendizajes de nuestro modelo solo se hagan sobre los datos de Train. Nos quedamos con el 80% de los datos para aprender y un 20% de los datos para comprobar el aprendizaje.

# 6) EDA

Ahora vamos a realizar una análisis exploratorio de datos, pero únicamente lo haremos sobre los datos de train, para evitar el data leakage.