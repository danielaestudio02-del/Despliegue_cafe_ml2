from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/clean_cafe_saless.csv")
MODEL_PATH = Path("model.joblib")

NUMERIC_FEATURES = ["Quantity", "Price Per Unit", "Total Spent"]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def describe_clusters(df: pd.DataFrame, labels) -> dict:
    df_clustered = df.copy()
    df_clustered["cluster"] = labels
    profiles = df_clustered.groupby("cluster")[NUMERIC_FEATURES].mean().round(2)
    overall = df[NUMERIC_FEATURES].mean()

    def nivel(valor, media, femenino=False):
        if valor >= media:
            return "alta" if femenino else "alto"
        return "baja" if femenino else "bajo"

    descriptions = {}
    for c, row in profiles.iterrows():
        descriptions[int(c)] = (
            f"Cantidad {nivel(row['Quantity'], overall['Quantity'], femenino=True)} "
            f"(~{row['Quantity']:.1f} uds), "
            f"precio unitario {nivel(row['Price Per Unit'], overall['Price Per Unit'])} "
            f"(~${row['Price Per Unit']:.2f}), "
            f"gasto total {nivel(row['Total Spent'], overall['Total Spent'])} "
            f"(~${row['Total Spent']:.2f})"
        )
    print(profiles)
    return descriptions


def train_model(df: pd.DataFrame) -> dict:
    # 1. Definir variables
    features = NUMERIC_FEATURES
    X = df[features].copy()

    # 2. Construir preprocesamiento + modelo
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("cluster", KMeans(n_clusters=3, random_state=42, n_init="auto")),
    ])

    # 3. Entrenar
    labels = model.fit_predict(X)

    # 4. Evaluar
    score = silhouette_score(model[:-1].transform(X), labels)
    print(f"Silhouette: {score:.3f}")

    # 5. Construir artifact
    return {
        "model": model,
        "features": features,
        "numeric_features": NUMERIC_FEATURES,
        "task": "clustering",
        "cluster_descriptions": describe_clusters(df, labels),
    }


def main() -> None:
    df = load_data()
    artifact = train_model(df)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
