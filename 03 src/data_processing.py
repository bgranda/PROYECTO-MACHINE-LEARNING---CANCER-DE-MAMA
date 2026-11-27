# Ruta corta para cargar datos, si estás en una carpeta y tienes que salir de ella y entrar en otra
file = "../01 data/01 raw/Breast_Cancer.csv"

# Por si alguna columna trae espacios
df.columns = df.columns.str.strip()

# Renombrar columnas, si quieres mantenerlo en la columna original hay que poner inplace=True
df.rename(columns={'Reginol Node Positive':'Regional Node Positive'}, inplace=True)

# Cambiar unos valores determinados dentro de una columna
df.loc[df['Grade'] == ' anaplastic; Grade IV', 'Grade'] = '4'

# Guardar el dataset en un archivo csv
df_processed = df.copy()
df_processed.to_csv('../01 data/02 proceseed/df_processed.csv', index=False)

# Comprobar el directorio donde se ha guardado el csv
print(os.listdir("../01 data/02 processeed"))

# Dividir en Train y Test
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survival Months', axis=1), df['Survival Months'], test_size=0.2, random_state=42)

# Percentiles
percentiles = y_train.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

# Detección outliers
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)

IQR = Q3 - Q1

lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

outliers = y_train[
    (y_train < lim_inf) |
    (y_train > lim_sup)
]

print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Límite inferior: {lim_inf:.2f}")
print(f"Límite superior: {lim_sup:.2f}")

print(f"\nNúmero de outliers: {len(outliers)}")
print(f"Porcentaje de outliers: {100 * len(outliers) / len(y_train):.2f}%")

# Evaluación normalidad: gráfica QQPlot
stats.probplot(
    y_train,
    dist="norm",
    plot=plt
)

plt.title("QQ Plot")
plt.show()

# Evaluación de normalidad: Test Shapiro-Wilk

from scipy.stats import shapiro

sample = y_train.sample(
    min(5000, len(y_train)),
    random_state=42
)

stat, p = shapiro(sample)

print("Estadístico:", stat)
print("p-value:", p)

if p > 0.05:
    print("No se rechaza la hipótesis de normalidad")
else:
    print("Se rechaza la hipótesis de normalidad")

# Transformaciones para conseguir distribución normal: Logaritmica, Box-cox, Yeo-Johnson

y_train_log = np.log1p(y_train)
print("Variable original")
print("Asimetría:", y_train.skew())
print("Curtosis:", y_train.kurtosis())

print("\nTransformación log1p")
print("Asimetría:", y_train_log.skew())
print("Curtosis:", y_train_log.kurtosis())

y_train_boxcox, lambda_bc = boxcox(y_train)
print("Lambda:", lambda_bc)
print("Asimetría:", pd.Series(y_train_boxcox).skew())
print("Curtosis:", pd.Series(y_train_boxcox).kurtosis())

print("Lambda Yeo-Johnson:", pt.lambdas_[0])
print("Asimetría:", stats.skew(y_yj))
print("Curtosis:", stats.kurtosis(y_yj))

# Punto de Referencia: Baseline

X_dummy = np.zeros((len(y_train), 1))
cv = KFold(n_splits=5, shuffle=True, random_state=42)

filas = {}
for estrategia in ["mean", "median"]:
    res = cross_validate(
        DummyRegressor(strategy=estrategia), X_dummy, y_train, cv=cv,
        scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"},
    )
    filas[estrategia] = {"MAE": -res["test_MAE"].mean(), "RMSE": -res["test_RMSE"].mean()}

baseline = pd.DataFrame(filas).T.round(2)
mae_dummy = baseline.loc["mean", "MAE"]
baseline

# Reporte de variables:

def data_report(df):
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)

    return concatenado.T
        
data_report(X_train)

# Correlación Pearson (variables numéricas solo)
train_df = X_train.copy()
train_df['Survival Months'] = y_train

corr_target = (train_df.corr(numeric_only=True)['Survival Months'].sort_values(ascending=False))
print(corr_target)

# Variables más correlacionadas (variables numéricas solo)

corr_target.drop('Survival Months').sort_values(key=abs, ascending=False)

# Correlación de Spearman (por orden) y el gráfico por tramos detectan otras formas.
corr_spearman = (train_df.corr(method='spearman', numeric_only=True)['Survival Months'].drop('Survival Months'))
print("Correlación de Spearman con Survival Months:")
print(corr_spearman.sort_values(key=abs, ascending=False).round(3))

# Análisis de multicolinealidad mediante VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_num = X_train.select_dtypes(include='number')

vif = pd.DataFrame({
    'Variable': X_num.columns,
    'VIF': [
        variance_inflation_factor(X_num.values, i)
        for i in range(X_num.shape[1])
    ]
})

vif.sort_values('VIF', ascending=False)

# Estudio de variables altamente correlacionadas

corr_matrix = X_train.corr(numeric_only=True)

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr = [
    col
    for col in upper.columns
    if any(abs(upper[col]) > 0.7)
]

print("Variables candidatas a excluir:")
print(high_corr)

# Kruskal-Wallis: ¿cambian los meses entre las categorías? (eta²: cuánto cambian; <0,01 casi nada, 0,01-0,06 poco, >0,06 bastante)
filas = []
for col in features_cat:
    grupos = [g.values for _, g in y_train.groupby(X_train[col])]
    H, p_val = stats.kruskal(*grupos)
    k, n_tot = len(grupos), len(y_train)
    eta2 = (H - k + 1) / (n_tot - k)
    filas.append({'Variable': col, 'p-valor': p_val, 'eta²': eta2})

kw = pd.DataFrame(filas).set_index('Variable').sort_values('eta²', ascending=False)
kw['p-valor'] = kw['p-valor'].map(lambda v: f"{v:.1e}")
kw['eta²'] = kw['eta²'].round(3)
kw

# Phi-K: asociación de todas las variables (numéricas y categóricas) con el target y entre sí
df_train = pd.concat([X_train, y_train], axis=1)

phik_matrix = df_train.phik_matrix()

phik_target = (phik_matrix['Survival Months'].drop('Survival Months').sort_values(ascending=False))

print(phik_target)

# Parejas de predictoras muy asociadas entre sí (Phi-K >= 0,7)
phik_pred = phik_matrix.drop(index='Survival Months', columns='Survival Months')
cols_pred = list(phik_pred.columns)

pares_altos = []
for i, a in enumerate(cols_pred):
    for b in cols_pred[i + 1:]:
        if phik_pred.loc[a, b] >= 0.7:
            pares_altos.append((a, b, phik_pred.loc[a, b]))
pares_altos = sorted(pares_altos, key=lambda x: -x[2])

for a, b, v in pares_altos:
    print(f"{a} - {b}: {v:.2f}")

# Importancia de variables numéricas: Random Forest y SelectKBest

X_train_num = X_train.select_dtypes(include=np.number)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_num, y_train)

importancias = pd.DataFrame({'Feature': X_train_num.columns, 'Score': rf.feature_importances_})
importancias = importancias.sort_values(by='Score', ascending=False).reset_index(drop=True)
print(importancias)

X_train_num = X_train.select_dtypes(include=np.number)
print("Dimensiones originales:", X_train_num.shape)

sel = SelectKBest(score_func=f_regression, k=3)
X_new = sel.fit_transform(X_train_num, y_train)
print("Dimensiones tras selección:", X_new.shape)
print(sel.scores_)

scores_df = pd.DataFrame({'Variable': X_train_num.columns, 'Score': sel.scores_})
scores_df = scores_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
print(scores_df)

# ¿Tiene el Random Forest señal real? Validación cruzada solo con train, frente al punto de referencia
cat_cols = X_train.select_dtypes(exclude='number').columns.to_list()

preproceso = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)], remainder='passthrough')

    # Modelo de referencia (predice siempre la media)
modelo_dummy = Pipeline([('prep', preproceso), ('dummy', DummyRegressor(strategy='mean'))])

res_dummy = cross_validate(
    modelo_dummy,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "R2": "r2"
    }
)

mae_dummy = -res_dummy["test_MAE"].mean()
r2_dummy = res_dummy["test_R2"].mean()

# Random Forest
modelo_rf = Pipeline([
    ('prep', preproceso),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )),
])

res_rf = cross_validate(
    modelo_rf,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "R2": "r2"
    }
)

mae_rf = -res_rf["test_MAE"].mean()
r2_rf = res_rf["test_R2"].mean()

print(f"Referencia (responder la media): MAE = {mae_dummy:.2f} meses | R² = {r2_dummy:.3f}")
print(f"Random Forest con todas las variables: MAE = {mae_rf:.2f} meses | R² = {r2_rf:.3f}")

# Importancia de todas las variables: Random Forest clásico y permutación
    # Codificación de variables categóricas
X_train_dummies = pd.get_dummies(X_train, drop_first=False)

    # Entrenamiento
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

rf.fit(X_train_dummies, y_train)

    # Importancias
importancias = pd.DataFrame({'Variable': X_train_dummies.columns, 'Importancia': rf.feature_importances_})

importancias = importancias.sort_values(by='Importancia', ascending=False)

    # Agrupamos por las categorías originales

resultados = []

for var in X_train.columns:

    columnas = [
        col for col in X_train_dummies.columns
        if col == var or col.startswith(var + "_")
    ]

    importancia = importancias.loc[
        importancias['Variable'].isin(columnas),
        'Importancia'
    ].sum()

    resultados.append([var, importancia])

resultado_final = pd.DataFrame(
    resultados,
    columns=["Variable", "Importancia"]
)

resultado_final = resultado_final.sort_values(
    by="Importancia",
    ascending=False
)

print(resultado_final)

# Importancia por permutación: se baraja cada variable y se mira cuánto empeora la predicción.
    # Se mide en un trozo de train que el modelo no ha visto (el test no se toca).

from sklearn.inspection import permutation_importance

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

modelo_rf.fit(X_tr, y_tr)

perm = permutation_importance(modelo_rf, X_val, y_val, n_repeats=10, random_state=42, scoring='r2', n_jobs=-1)

imp_perm = pd.DataFrame({
    'Variable': X_val.columns,
    'Caída de R²': perm.importances_mean,
    'Desviación': perm.importances_std,
}).sort_values('Caída de R²', ascending=False).reset_index(drop=True)

imp_perm.round(4)

# Comparación de los dos métodos (posición en cada ranking)
comparacion = pd.DataFrame({
    'Puesto Random Forest clásico': resultado_final.set_index('Variable')['Importancia'].rank(ascending=False).astype(int),
    'Puesto permutación': imp_perm.set_index('Variable')['Caída de R²'].rank(ascending=False).astype(int),
}).sort_values('Puesto permutación')
comparacion