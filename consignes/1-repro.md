# Introduction — la pyramide de reproductibilité

Reproduire un entraînement, c'est obtenir les mêmes résultats à partir des mêmes inputs — quelle que soit la machine et quel que soit le moment. Aucun réglage isolé n'y suffit : la reproductibilité se construit par couches, chacune fermant une source de variation que la précédente laissait passer. On parle de **pyramide de reproductibilité**.

```mermaid
block-beta
  columns 11
  space:5 r["6. Registry"]:1 space:5
  space:4 p["5. MLproject"]:3 space:4
  space:3 t["4. Tracking"]:5 space:3
  space:2 e["3. Docker"]:7 space:2
  space:1 d["2. uv.lock"]:9 space:1
  s["1. Seed"]:11
```

De la base au sommet, chaque couche s'appuie sur celles du dessous :

1. **Seed** — fixe l'aléatoire (`RANDOM_STATE`) : déterminisme sur une même machine.
2. **uv.lock** — fige les versions exactes des packages Python.
3. **Docker** — fige l'OS, l'architecture et les libs natives (BLAS, OpenMP).
4. **Tracking** — logge chaque run avec MLflow : params, métriques, dataset, modèle.
5. **MLproject** — empaquette l'invocation : entry points et paramètres déclarés.
6. **Registry** — versionne les modèles et gère les alias (`@champion`).

On va construire cette pyramide progressivement : chaque partie ci-dessous ajoute une couche, de la base au sommet.

# 1 - Seed

Vous devriez voir un script d'entraînement [train.py](src/scripts/train.py).

**1. lancer la commande deux fois et observer les résultats**

```bash
uv sync
uv run python src/scripts/train.py
uv run python src/scripts/train.py
```
En comparant les sorties des exécutions, vous devriez observer des résultats différents.


**2. Importer et appeler `fix_all_seeds()` dans `src/scripts/train.py`**

Dans le projet vous pouvez trouver le fichier [seeds.py](src/utils/seeds.py) : A l'intérieur la fonction `fix_all_seeds()` (qui fixe `random`, `numpy` et `PYTHONHASHSEED`) et la constante `RANDOM_STATE`. Il suffit de l'appeler en haut du fichier, avant toute autre logique :

```python
from utils.seeds import fix_all_seeds, RANDOM_STATE

fix_all_seeds()
```

**3. Brancher `RANDOM_STATE` partout où sklearn accepte un seed**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = make_pipeline(
    pre,
    MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, solver='lbfgs', random_state=RANDOM_STATE)
)
```

Maintenant si l'on exécute successivement deux fois le script [train.py](src/scripts/train.py) on devrait observer les mêmes résultats.
```bash
uv run python src/scripts/train.py
uv run python src/scripts/train.py
```

# 2 - Dependencies

Mêmes seeds, même code, mêmes données… mais si les versions de `numpy` ou `scikit-learn` changent entre deux machines (ou entre deux installations à quelques mois d'intervalle) le comportement numérique peut bouger et les résultats différer à nouveau. Les seeds ne suffisent donc pas : il faut aussi **figer les versions des dépendances**.

C'est le rôle du **lock file**. `uv lock` résout les dépendances une fois et fige les versions exactes dans le `uv.lock`. Ensuite `uv sync` réinstalle précisément ces versions sur n'importe quelle machine, tant qu'on ne relance pas `uv lock --upgrade`. Comme `uv.lock` est committé dans le dépôt, c'est lui qui garantit que tout le monde entraîne avec les mêmes dépendances.

En complément, `exclude-newer` dans `pyproject.toml` fixe une borne temporelle de résolution :

```toml
[tool.uv]
exclude-newer = "2026-06-01T00:00:00Z"
```

`uv` refuse alors de résoudre des packages publiés après cette date : on contrôle *quand* on adopte de nouvelles versions plutôt que de les subir, ce qui évite aussi d'introduire silencieusement une version vulnérable fraîchement publiée. Monter de version devient un acte explicite (repousser la date puis relancer `uv lock --upgrade`) et non un effet de bord subi.


# 3 - Environment

**1. Réflexion — pourquoi `uv.lock` ne suffit pas**

`uv.lock` figure exactement les versions des packages Python. Mais le binaire `numpy` linke des libs natives (BLAS, OpenMP) qui dépendent de l'OS et de l'architecture. Sur deux machines différentes, on peut donc avoir le même `uv.lock` et toujours des résultats légèrement différents.



**2. Créer un `Dockerfile.train` à la racine**


```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "src/scripts/train.py", "--data", "https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet"]
```

**3. Build & run sur deux architectures**

```bash
docker build -f Dockerfile.train --platform linux/amd64 -t train-churn .
docker run --rm --platform linux/amd64 train-churn
```

```bash
docker build -f Dockerfile.train --platform linux/arm64 -t train-churn .
docker run --rm --platform linux/arm64 train-churn
```

Vous devriez observer des résultats différents.

Selon l'architecture, certaines libs (comme `numpy`) peuvent s'appuyer sur des implémentations natives différentes et produire des résultats légèrement différents. Pour une vraie reproductibilité bout-en-bout, on figerait une seule architecture cible (généralement `linux/amd64` en prod).

Remarque : même avec tout ça, un changement subtil peut passer. La parade en prod : tests automatisés qui rejouent un échantillon fixe en CI et comparent les sorties à une baseline.

# 4 - Experiment Tracking

Notre entraînement est maintenant déterministe. Mais "mêmes inputs" suppose qu'on sache lesquels : quel snapshot de `churn.parquet`, quels hyperparamètres, et quelles métriques ça a produit.
C'est ici qu'intervient MLflow. L'experiment tracking enregistre, à chaque run d'entraînement, ses paramètres, métriques, dataset et modèle. Très utile pour comparer les runs et retrouver ce qui a produit un modèle donné.



**1. Démarrer le tracking server**

MLflow met à disposition un serveur de tracking : c'est lui qui reçoit et stocke les runs (params, métriques, modèles) et expose l'UI pour les explorer.

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts --artifacts-destination ./mlartifacts \
  --allowed-hosts '*' --cors-allowed-origins '*'
```

```bash
export MLFLOW_TRACKING_URI=https://<votre-mlflow>
curl -s "$MLFLOW_TRACKING_URI/health"
```

**2. Premier contact : `autolog()`**

Le plus rapide pour voir un run apparaître dans l'UI, c'est une seule ligne. Ajoutez `argparse` et `autolog()` à `src/scripts/train.py` :

```python
import mlflow
import argparse

from utils.seeds import fix_all_seeds, RANDOM_STATE

mlflow.set_experiment("Churn Predictor")
mlflow.sklearn.autolog()
fix_all_seeds()


def train(data_path):
    df = pd.read_parquet(data_path)
    # ... corps inchangé ...
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input parquet file")
    args = parser.parse_args()
    train(args.data)
```

Lancez l'entraînement :

```bash
uv run python src/scripts/train.py --data https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet
```

Ouvrez mlflow dans le navigateur : votre run vient d'apparaître. Prenez le temps d'explorer le serveur. C'est là que vivra tout ce qu'on va logger ensuite.

Un serveur MLflow, c'est deux "stores" :
- un **backend store** pour les métadonnées des runs (params, métriques, tags)
- un **artifact store** pour les fichiers (modèles, datasets).


Dans la UI vous pouvez explorer :

- **Liste des experiments** : un experiment regroupe des runs. Vous voyez "Churn Predictor". `Default` est l'experiment utilisé quand aucun n'est précisé.
- **Table des runs** : un entraînement = une ligne. Triez par métrique, ajoutez/retirez des colonnes (params, métriques), et testez la barre de recherche : `metrics.auc > 0.8`, `params.solver = 'lbfgs'`.
- **Détail d'un run** :
  - *Overview* : run ID, statut, durée, source, params et métriques.
  - *Artifacts* : le dossier `model/` loggé par `autolog` (nom par défaut). Ouvrez-le : `MLmodel` décrit le modèle, `requirements.txt` / `python_env.yaml` capturent l'environnement, `model.pkl` est le modèle sérialisé.
  - *Model metrics*, *System metrics*, *Tags* : le reste des métadonnées d'exécution.
- **Comparer des runs** : relancez l'entraînement (seeds fixés ⇒ mêmes métriques), cochez deux runs, puis **Compare** : vue côte à côte et coordonnées parallèles, l'outil pour juger un changement d'hyperparamètres.

Tout ça est aussi accessible en CLI (pratique en CI ou sans navigateur) :

```bash
uv run mlflow experiments search
uv run mlflow runs list --experiment-id <id>
```

`autolog()` est rapide à mettre en place mais opaque : on ne contrôle pas finement ce qui est loggé. MLflow nous permet de tout faire manuellement.


**3. Retirer `autolog` et wrapper le training dans un run**

On veut maîtriser quels params, quelles métriques, quel dataset, quel modèle et obtenir un vrai lineage entre l'entraînement et l'inférence.

```python
def train(data_path):
    df = pd.read_parquet(data_path)
    # ...

    with mlflow.start_run():
        # toute la logique de training va ici
        ...
```

**4. Centraliser les hyperparamètres**

```python
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=RANDOM_STATE
)

params = {
    "hidden_layer_sizes": (50,),
    "max_iter": 100,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

with mlflow.start_run():
    mlflow.log_params({**params, "test_size": test_size})

    pre = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["plan"]),
        remainder="passthrough",
    )
    model = make_pipeline(pre, MLPClassifier(**params))
    model.fit(X_train, y_train)
```

Le dictionnaire `params` est utilisé deux fois : pour instancier le modèle ET pour logger (impossible que les deux dérivent).

**5. Logger les métriques**

```python
auc = roc_auc_score(y_test, probs[:, 1])
loss = log_loss(y_test, probs)
total_predicted_churn = int(preds.sum())

mlflow.log_metric("auc", auc)
mlflow.log_metric("log_loss", loss)
mlflow.log_metric("total_predicted_churn", total_predicted_churn)
```

**6. Logger le dataset (lineage)**

Tout en haut de `train()`, juste après la lecture du parquet :

```python
df = pd.read_parquet(data_path)
dataset = mlflow.data.from_pandas(df, source=data_path, name="churn-dataset")
```

Puis dans le `with mlflow.start_run():` :

```python
mlflow.log_input(dataset, context="training")
```

**7. Logger le modèle**

```python
mlflow.sklearn.log_model(sk_model=model, artifact_path="churn_model")
```


**8. Mettre à jour `src/scripts/infer.py` pour charger depuis MLflow**

```python
import argparse
from pathlib import Path

import mlflow
import pandas as pd


def infer(data_path, run_id, output_path):
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["churned"], errors="ignore")

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/churn_model")
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["prediction"] = preds
    out["churn_probability"] = probs

    with mlflow.start_run():
        input_dataset = mlflow.data.from_pandas(df, source=data_path, name="churn-input")
        mlflow.log_input(input_dataset, context="inference")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path)
        mlflow.log_artifact(output_path)

        print(f"Predictions written to {output_path} and logged to MLflow ({len(out)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-id", required=True, help="MLflow run ID containing the logged model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    infer(args.data, args.run_id, args.output)
```

**9. Workflow complet**

```bash
uv run python src/scripts/train.py --data https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet

uv run python src/scripts/infer.py --data https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet --run-id <id> --output data/predictions.parquet
```

Dans l'UI, le run d'inference doit afficher :
- Datasets used : `churn-input` (context: `inference`)
- Artifacts : `predictions.parquet`

Et le run de d'entraînement est lié au modèle chargé.

---


# 5 - MLflow Project

Jusqu'ici, lancer un entraînement suppose de connaître la bonne commande : quel script, quels arguments, quelles valeurs par défaut. Cette connaissance vit dans la tête de celui qui lance, pas dans le projet.

Un MLflow Project rend cette invocation explicite : n'importe qui — ou une CI — peut alors lancer `mlflow run . -e train -P ...` sans rien savoir de la commande sous-jacente. C'est la dernière couche de reproductibilité : après le code, l'environnement et l'exécution déterministes, on fige aussi *la façon de lancer*.

C'est aussi un premier pas vers l'automatisation : une fois l'invocation standardisée, elle devient le point d'entrée que peuvent appeler les orchestrateurs et la CI/CD (Kubeflow Pipelines, GitHub Actions).


**1. Créer le fichier `MLproject` à la racine**

Le fichier `MLproject` déclare des *entry points* nommés (ici `train` et `infer`), chacun avec ses paramètres, leurs valeurs par défaut et la commande à exécuter. C'est ce qui permet de lancer un entraînement ou une inférence via `mlflow run` sans connaître la ligne de commande exacte.

```yaml
name: Churn Predictor

entry_points:
    train:
        parameters:
            data: {type: string, default: "https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet"}
        command: "python src/scripts/train.py --data {data}"
    infer:
        parameters:
            data: {type: string, default: "https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet"}
            run_id: {type: string}
            output: {type: string, default: data/predictions.parquet}
        command: "python src/scripts/infer.py --data {data} --run-id {run_id} --output {output}"
```

**2. Lancer l'entraînement via MLflow**

```bash
uv run mlflow run . -e train --env-manager local --experiment-name "Churn Predictor"
```

**3. Inspecter dans l'interface**

Ouvrez l'interface MLflow. Le run apparaît dans l'experiment "Churn Predictor", exactement comme avec `python src/scripts/train.py` mais cette fois l'invocation est packagée et reproductible.

**4. Lancer l'inférence via MLflow**

```bash
uv run mlflow run . -e infer --env-manager local -P run_id=<id>
```

Dans l'UI, le run d'inference doit afficher les datasets et artifacts attendus, et rester lié au run de training source.

---


# 6 - Registry

Travailler avec un `run_id` brut a ses limites :

- Pas humainement mémorisable
- Pas de notion de version (v1, v2, v3)
- Pas de canal "champion" / "production" / "staging"
- Difficile à promouvoir/rollback proprement

Le Model Registry résout ça : il versionne les modèles enregistrés et permet de leur attacher des alias (`@champion`, `@staging`, …) pour découpler le code qui charge un modèle de la version exacte servie.

**1. Enregistrer le modèle au registry**

Dans `src/scripts/train.py`, modifier l'appel à `log_model` :

```python
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="churn_model",
    registered_model_name="churn-model",
)
```

À la première exécution, MLflow crée le registered model `churn-model` avec une Version 1. À chaque exécution suivante, une nouvelle version (2, 3, …) est ajoutée automatiquement.

**2. Ré-entraîner et inspecter dans l'UI**

```bash
uv run mlflow run . -e train --env-manager local
```

Ouvrez la UI puis l'onglet Models dans la barre latérale. Vous voyez `churn-model` avec sa Version 1, et un lien vers le run source.

**3. Charger depuis le registry dans `src/scripts/infer.py`**

```python
def infer(data_path, model_name, version, output_path):
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["churned"], errors="ignore")

    model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    # ... reste inchangé ...
```

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-name", required=True, help="Registered model name")
    parser.add_argument("--version", required=True, help="Registered model version (e.g. 1, 2, ...)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    infer(args.data, args.model_name, args.version, args.output)
```

**4. Mettre à jour le `MLproject`**

```yaml
infer:
    parameters:
        data: {type: string, default: "https://raw.githubusercontent.com/VersoML/mlops-training-data/main/churn.parquet"}
        model_name: {type: string, default: churn-model}
        version: {type: string, default: "1"}
        output: {type: string, default: data/predictions.parquet}
    command: "python src/scripts/infer.py --data {data} --model-name {model_name} --version {version} --output {output}"
```

**5. Créer une seconde version**

```bash
uv run mlflow run . -e train --env-manager local
```

Dans l'UI, `churn-model` a maintenant Version 1 ET Version 2.

**6. Comparer les prédictions entre versions**

```bash
uv run mlflow run . -e infer --env-manager local -P version=1
uv run mlflow run . -e infer --env-manager local -P version=2
```

Les deux runs d'inference apparaissent dans l'UI. Vous pouvez comparer les datasets et artifacts.

**7. Aliases : la vraie pratique en prod**

Plutôt que de coder en dur `version=2` partout, on assigne un alias dans l'interface (par ex. `champion` sur la Version 2) et on charge :

```python
mlflow.sklearn.load_model("models:/churn-model@champion")
```

Pour promouvoir la Version 3 quand elle est prête, il suffit de réassigner l'alias `champion` dans l'UI, sans modifier le code de l'inference.


---

## Est-ce que c'est suffisant ?

Malgré toutes les barrières (seeds, lock, image Docker, MLflow), un détail peut toujours nous échapper. 

Pour un système robuste en prod :

- Tests de reproductibilité automatisés en CI : rejouer un sample fixe et comparer la sortie à un baseline figé.
- Diff de modèles : pour deux runs censées être équivalents, comparer les coefficients/poids et alerter si un seuil est dépassé.
- Surveillance des données : vérifier en production que la distribution des inputs ne dérive pas (sinon les prédictions divergent même avec un modèle figé).
