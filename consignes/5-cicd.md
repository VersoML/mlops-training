# Introduction : tester et garder automatiquement

On a un modèle reproductible (module 1), servi (2), orchestré (3), observé (4). Il manque le fil qui, à **chaque changement**, vérifie que tout tient encore, sans qu'un humain y pense.

Sans ce fil : un collègue ouvre une PR qui touche un hyperparamètre ou le prétraitement. Qui garantit que le schéma des données tient toujours, que les prédictions du serveur n'ont pas changé en silence, que les pipelines compilent encore ? À la main, personne ne le fait avant la prod.

C'est le rôle de la **CI** (Continuous Integration) : à chaque push / PR, une machine rejoue automatiquement une batterie de vérifications, et bloque la PR si l'une échoue.

Particularité du ML : valider le **code** ne suffit pas, il faut **aussi** valider les **données** et le **service**. Et ce qui juge la **qualité du modèle** ne vit pas dans la CI de PR du tout, on verra pourquoi (§5).

# 1 - Les vérifications qui composeront la CI

Avant tout workflow, on étudie les vérifications qui formeront la CI : pour chacune, la **commande qui la lance** puis le **code** qui la porte. Les outils viennent du groupe `dev` de `pyproject.toml` :

```toml
[dependency-groups]
dev = ["pytest>=8.0.0", "httpx>=0.27.0", "ruff>=0.6.0", "mypy>=1.11.0"]
```

Quatre outils, présentés avant de s'en servir :

- **ruff** : linter + formateur Python.
- **mypy** : *type checker* statique.
- **pytest** : le lanceur de tests Python. Il découvre les fonctions `test_`*, exécute chacune, et rapporte les échecs.
- **httpx** : client HTTP sur lequel s'appuie le `TestClient` de FastAPI.

## A. Vérifier le code

**1. ruff (lint + format).** ruff est un **linter** et **formateur** Python écrit en Rust, très rapide. Il remplace à lui seul flake8 + isort + black + pyupgrade.

```bash
uv run ruff check tests/ src/utils/schema.py src/utils/model.py
uv run ruff format --check tests/ src/utils/schema.py src/utils/model.py
```

La première commande **lint** : imports inutilisés, noms non définis, motifs propices aux bugs (règles `E`, `F`, `I`, `UP`, `B`), tri des imports. La seconde vérifie le **formatage** et échoue si un fichier n'est pas formaté, sans rien réécrire. On le **cible** sur la surface introduite par ce module : reformater les modules 1-4/6 serait du bruit. Config dans `[tool.ruff]` de `pyproject.toml`.

**2. mypy (vérification de types).** mypy lit les annotations de type et signale les incohérences **avant** l'exécution (un `str` passé là où un `int` est attendu, un `None` oublié).

```bash
uv run mypy
```

Ciblé sur `src/serving/server.py`, la surface de **contrat** du service (config `[tool.mypy]`). **Non bloquant** au début (`continue-on-error`) : strict d'emblée, il rejetterait la PR de mise en place à cause des stubs de types absents de `mlflow`/`sklearn`/`evidently`/`torch`.

**3. Import sanity, le test le moins cher.**

```bash
uv run python -c "import serving.server, orchestration.components.training"
```

Grâce à l'install editable de uv (`[tool.setuptools.packages.find] where=["src"]`, `namespaces=true`), les packages s'importent **comme en prod** (`uvicorn serving.server:app`), sans bricoler `PYTHONPATH`. Si un import casse (renommage, dépendance oubliée), on le sait avant tout test plus lourd.

**4. Compiler les pipelines KFP, sans cluster** (`tests/compile_pipelines.py`).

```bash
uv run python tests/compile_pipelines.py
```

Compiler le DSL en YAML attrape les erreurs de graphe (mauvais nom de sortie, type d'artefact incompatible) sans déployer. **Le piège** : chaque pipeline fait `from orchestration.client import kfp_client`, et `client.py` s'authentifie à Dex **à l'import** (module 3). Un import en CI planterait. On injecte donc un faux `orchestration.client` dans `sys.modules` **avant** d'importer les pipelines. Le vrai `client.py` ne s'exécute jamais, le DSL est compilé pour de vrai :

```python
import sys
import tempfile
import types
from pathlib import Path

_stub = types.ModuleType("orchestration.client")
_stub.kfp_client = None
_stub.get_kfp_client = lambda *a, **k: None
sys.modules["orchestration.client"] = _stub

from kfp.compiler import Compiler  # noqa: E402

from orchestration.pipelines.inference import inference_pipeline  # noqa: E402
from orchestration.pipelines.monitoring import monitoring_pipeline  # noqa: E402
from orchestration.pipelines.retraining import (  # noqa: E402
    drift_triggered_retraining_pipeline,
)
from orchestration.pipelines.training import training_pipeline  # noqa: E402

PIPELINES = {
    "training": training_pipeline,
    "inference": inference_pipeline,
    "retraining": drift_triggered_retraining_pipeline,
    "monitoring": monitoring_pipeline,
}


def main() -> int:
    out = Path(tempfile.mkdtemp(prefix="kfp-compile-"))
    for name, fn in PIPELINES.items():
        Compiler().compile(fn, str(out / f"{name}.yaml"))
        print(f"OK {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**5. Tests de contrat de l'API** (`tests/test_server.py`).

```bash
uv run pytest tests/test_server.py
```

Le test exerce l'app FastAPI via `TestClient` (qui s'appuie sur httpx). Le serveur charge un modèle MLflow au démarrage : on **monkeypatch** `mlflow.pyfunc.load_model` par un faux modèle **avant** que le `lifespan` ne s'exécute, pour ne jamais toucher un registry réel. On couvre `/health`, `/predict`, les rejets pydantic 422, `/predict/batch` vide → 400, et `/capture/flush` :

```python
import pandas as pd
import pytest
from fastapi.testclient import TestClient

VALID = {
    "tenure_days": 200,
    "nb_logins_30j": 12,
    "nb_features_used": 4,
    "plan": "Pro",
    "support_tickets_90j": 1,
    "mrr_eur": 49.0,
    "has_integration": True,
}


class _FakeModel:
    def predict(self, df: pd.DataFrame):
        return [True] * len(df)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "capture"))
    monkeypatch.setenv("CAPTURE_FLUSH_SIZE", "1000")

    import mlflow.pyfunc

    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: _FakeModel())

    from serving import server

    with TestClient(server.app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_happy_path(client):
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 200
    assert resp.json() == {"churned": True}


def test_predict_rejects_negative_int(client):
    bad = {**VALID, "tenure_days": -1}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_unknown_plan(client):
    bad = {**VALID, "plan": "Gold"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_empty_is_400(client):
    resp = client.post("/predict/batch", json={"instances": []})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "instances must not be empty"


def test_capture_flush_counts_rows(client):
    client.post("/predict", json=VALID)
    client.post("/predict/batch", json={"instances": [VALID, VALID]})
    resp = client.post("/capture/flush")
    assert resp.status_code == 200
    assert resp.json()["flushed_rows"] == 3
```

## B. Vérifier les données et le service

C'est le cœur MLOps. Déployer un nouveau code serveur ou une nouvelle pipeline, c'est risquer deux régressions que les vérifs du code ne voient pas : la **donnée** qui ne respecte plus le schéma, et le **résultat** du serveur qui dérive en silence.

**6. Contrat de données, pandera** (`src/utils/schema.py` + `tests/test_schema.py`).

```bash
uv run python -m utils.schema --data tests/fixtures/sample.parquet
uv run pytest tests/test_schema.py
```

pandera valide un DataFrame contre un **schéma** déclaré et lève si une règle est violée. En ML, une donnée hors-contrat ne plante pas : elle **dégrade silencieusement** le modèle. `schema.py` est la **source unique de vérité** des colonnes, partagée par pydantic (serving, module 2) et la `DataDefinition` Evidently (module 4) :

```python
import argparse
import sys

import pandas as pd
import pandera.pandas as pa

PLANS = ["Free", "Pro", "Enterprise"]

churn_schema = pa.DataFrameSchema(
    {
        "tenure_days": pa.Column(int, pa.Check.ge(0)),
        "nb_logins_30j": pa.Column(int, pa.Check.ge(0)),
        "nb_features_used": pa.Column(int, pa.Check.ge(0)),
        "plan": pa.Column(str, pa.Check.isin(PLANS)),
        "support_tickets_90j": pa.Column(int, pa.Check.ge(0)),
        "mrr_eur": pa.Column(float, pa.Check.ge(0)),
        "has_integration": pa.Column(bool),
        "churned": pa.Column(bool),
    }
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    return churn_schema.validate(df, lazy=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Valide un parquet churn contre le contrat.")
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    try:
        validate(df)
    except pa.errors.SchemaErrors as exc:
        print(exc.failure_cases.to_string(index=False), file=sys.stderr)
        return 1
    print(f"{len(df)} lignes conformes au contrat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Le test vérifie que le fixture committé passe et qu'une donnée corrompue (plan inconnu, compteur négatif) est **rejetée** :

```python
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from utils.schema import validate

FIXTURE = Path(__file__).parent / "fixtures" / "sample.parquet"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


def test_fixture_satisfies_contract(df):
    validate(df)


def test_unknown_plan_is_rejected(df):
    bad = df.copy()
    bad.loc[bad.index[0], "plan"] = "Gold"
    with pytest.raises(pa.errors.SchemaErrors):
        validate(bad)


def test_negative_counter_is_rejected(df):
    bad = df.copy()
    bad.loc[bad.index[0], "support_tickets_90j"] = -1
    with pytest.raises(pa.errors.SchemaErrors):
        validate(bad)
```

**7. Prédictions reproductibles du serveur, golden** (`tests/test_server_golden.py`).

```bash
uv run pytest tests/test_server_golden.py
```

Un test *golden* (de **caractérisation**) fige la sortie observable dans un fichier de référence et échoue si elle change. Pour des entrées figées, le serveur doit renvoyer toujours les mêmes prédictions. Le modèle est reconstruit **déterministe** via `build_model`, la **même factory** que `train.py` (`src/utils/model.py`) :

```python
from sklearn.compose import make_column_transformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from utils.seeds import RANDOM_STATE

PARAMS = {
    "hidden_layer_sizes": (50,),
    "max_iter": 100,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}


def build_model():
    pre = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), ["plan"]),
        remainder="passthrough",
    )
    return make_pipeline(pre, MLPClassifier(**PARAMS))
```

On asserte les **classes** (`bool`), pas les probabilités : la proba d'un MLP dépend de la BLAS (donc de l'archi), la classe non si la proba est franche. Le golden ne retient donc que des lignes à **marge > 0.10** (loin de 0.5), pour rester stable cross-plateforme. Il est régénéré par `tests/_generate_golden.py` :

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.model import build_model
from utils.seeds import fix_all_seeds

HERE = Path(__file__).parent
FIXTURE = HERE / "fixtures" / "sample.parquet"
GOLDEN = HERE / "golden" / "server_predictions.json"
MARGIN = 0.10


def main() -> None:
    fix_all_seeds()
    df = pd.read_parquet(FIXTURE)
    X = df.drop(columns=["churned"])
    model = build_model()
    model.fit(X, df["churned"])

    proba = model.predict_proba(X)[:, 1]
    confident = np.abs(proba - 0.5) > MARGIN
    idx = sorted(int(i) for i in np.where(confident)[0][:20])

    sample = X.iloc[idx]
    margin = float(np.abs(proba[idx] - 0.5).min())
    assert margin > MARGIN, "classe non stable cross-arch"

    preds = model.predict(sample)
    records = json.loads(sample.to_json(orient="records"))
    golden = [{"input": r, "churned": bool(p)} for r, p in zip(records, preds, strict=True)]
    GOLDEN.write_text(json.dumps(golden, indent=2))


if __name__ == "__main__":
    main()
```

Le test rejoue ces entrées contre l'app FastAPI (modèle monkeypatché à la place du registry) et compare les `churned` :

```python
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from utils.model import build_model
from utils.seeds import fix_all_seeds

HERE = Path(__file__).parent
FIXTURE = HERE / "fixtures" / "sample.parquet"
GOLDEN = HERE / "golden" / "server_predictions.json"


def _fitted_model():
    fix_all_seeds()
    df = pd.read_parquet(FIXTURE)
    model = build_model()
    model.fit(df.drop(columns=["churned"]), df["churned"])
    return model


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "capture"))
    monkeypatch.setenv("CAPTURE_FLUSH_SIZE", "100000")

    import mlflow.pyfunc

    model = _fitted_model()
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: model)

    from serving import server

    with TestClient(server.app) as c:
        yield c


def test_server_reproduces_golden_predictions(client):
    golden = json.loads(GOLDEN.read_text())
    resp = client.post("/predict/batch", json={"instances": [g["input"] for g in golden]})
    assert resp.status_code == 200
    got = [p["churned"] for p in resp.json()["predictions"]]
    assert got == [g["churned"] for g in golden]
```

Un changement de prétraitement, d'ordre de colonnes, de dtype, de sérialisation, ou un swap de modèle qui altère le contrat → rouge. (Le fixture committé est produit une fois par `tests/_generate_fixture.py`, qui réutilise le générateur synthétique du module 1.)

## C. Vérifier l'environnement (

**8. `uv sync --frozen`.** Installe les dépendances **exactement** depuis `uv.lock`, et échoue si `uv.lock` ne correspond plus à `pyproject.toml`. C'est la reproductibilité du module 1 transformée en vérification : pas de « ça marche sur ma machine ».



# 2 - Composer un workflow GitHub Actions

GitHub Actions exécute des **workflows** : des automatisations décrites en YAML, déclenchées par des événements du dépôt. Un workflow = un fichier dans `.github/workflows/`. On l'apprend ici pièce par pièce, sur un exemple minimal, le **seul** workflow fourni clé en main de ce module.

## L'exemple : un hello-world

`.github/workflows/hello.yaml` :

```yaml
name: hello
on:
  workflow_dispatch:
jobs:
  hello:
    runs-on: ubuntu-latest
    steps:
      - run: echo "Hello GitHub Actions"
      - name: Contexte
        run: echo "ref=${{ github.ref_name }}  commit=${{ github.sha }}  acteur=${{ github.actor }}"
```

## Anatomie, clé par clé

- `**name:**` le nom affiché du workflow dans l'onglet **Actions** du dépôt (cosmétique).
- `**on:`** le ou les **événements** déclencheurs, le cœur de l'automatisation :
  - `push` / `pull_request` à chaque push ou mise à jour de PR,
  - `workflow_dispatch` déclenchement **manuel** (bouton « Run workflow »), ce qu'utilise le hello-world,
  - `schedule` cron.
- `**jobs:`** un workflow contient un ou plusieurs **jobs**. Chaque clé sous `jobs:` est l'**identifiant** du job (`hello` ici). Les jobs tournent en **parallèle** par défaut.
- `**runs-on:`** le **runner**, la machine qui exécute le job. `ubuntu-latest` = un runner Linux fourni par GitHub, neuf et jetable à chaque run.
- `**steps:`** la liste **ordonnée** des étapes du job. Chaque `-` est un step, de l'un des deux types :
  - `**run:`** exécute une commande shell,
  - `**uses:**` invoque une **action** réutilisable (voir plus bas).
- `**name:` (sur un step)** libellé du step dans les logs (`- name: Contexte`), optionnel.
- `**${{ … }}`** une **expression**. GitHub y injecte le **contexte** : `${{ github.ref_name }}` (branche), `${{ github.sha }}` (commit), `${{ github.actor }}` (auteur), `${{ secrets.X }}` (un secret)…

## Les pièces de syntaxe en plus

De quoi composer les vraies gates.

`**uses:` + `with:`** une action de la Marketplace, paramétrée par `with:` :

```yaml
- uses: actions/checkout@v4
- uses: astral-sh/setup-uv@v6
  with:
    enable-cache: true
    cache-dependency-glob: uv.lock
```

`actions/checkout` est quasi toujours le **premier** step : le runner démarre **vide**, sans le code. `@v4` épingle la version de l'action.

`**run: |`** un step shell multi-lignes :

```yaml
- run: |
    echo "première commande"
    echo "deuxième commande"
```

`**needs:**` met des jobs **en série** : un job avec `needs: X` ne démarre que si `X` a réussi (*fail-fast*).

```yaml
jobs:
  premier:
    runs-on: ubuntu-latest
    steps:
      - run: echo "je tourne d'abord"
  second:
    needs: premier
    runs-on: ubuntu-latest
    steps:
      - run: echo "seulement si premier a réussi"
```

`**continue-on-error: true**` un step qui échoue **ne casse pas** le run (utile pour une vérif qu'on veut voir sans bloquer) :

```yaml
- run: uv run mypy
  continue-on-error: true
```

`**permissions:**` les droits du `GITHUB_TOKEN` du run. `contents: read` = lecture seule, le moindre privilège pour une CI qui ne fait que tester :

```yaml
permissions:
  contents: read
```

`**concurrency:**` regroupe les runs. `cancel-in-progress: true` annule le run précédent du même groupe (un nouveau push sur la même branche annule l'ancien, pour économiser le runner) :

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

**Filtre `paths:`** restreint un événement aux chemins pertinents :

```yaml
on:
  pull_request:
    paths:
      - "src/**"
  workflow_dispatch:
```

## act, rejouer en local

Pousser juste pour voir si un workflow passe (« push-and-pray ») est lent. `[act](https://github.com/nektos/act)` exécute les workflows sur la machine, dans des conteneurs Docker qui reproduisent le runner. On valide **avant** de pousser :

```bash
act -l
act workflow_dispatch -W .github/workflows/hello.yaml
```

(Pièges pour les workflows lourds : §6.)

# 3 - À vous : composer `ci.yaml` (CI logicielle)

Les vérifs du code sont prêtes (§1.A + §1.C) et la syntaxe est connue (§2). **À faire** : écrire `.github/workflows/ci.yaml` qui les rejoue à chaque PR et push sur `main`.

Le workflow doit :

- se déclencher sur `pull_request` **et** `push` (branche `main`)
- déclarer `permissions: contents: read` et une `concurrency` qui annule le run précédent de la même branche
- enchaîner **deux jobs en série** :
  - `quality` : ruff (`check` puis `format --check`), puis mypy en `continue-on-error`
  - `test` (`needs: quality`) : `uv sync --frozen`, import-sanity, compile KFP, tests de contrat (`test_server.py`)
- faire commencer chaque job par `actions/checkout` → `astral-sh/setup-uv` (cache sur `uv.lock`) → `uv python install 3.12`

Chaque step de vérification est une commande de §1, et la syntaxe pour les assembler est en §2. Rejouez le workflow en local avec act (§6) jusqu'au vert avant de pousser.

# 4 - À vous : composer `ml-ci.yaml` (données & service)

**À faire** : écrire `.github/workflows/ml-ci.yaml` pour les deux gates de §1.B, **auto-porté** (aucun cluster, aucun MLflow, aucun secret).

Le workflow doit :

- se déclencher sur `pull_request` **filtré par `paths:`** sur ce qui touche données ou service (`src/**`, `tests/**`, `pyproject.toml`, `uv.lock`), plus `workflow_dispatch`
- tenir en **un seul job** (`gates`)
- après les briques communes (`actions/checkout`, `astral-sh/setup-uv`, `uv python install 3.12`, `uv sync --frozen`), lancer le **contrat de données** (CLI `python -m utils.schema --data …` puis `pytest test_schema.py`) puis le **golden** (`pytest test_server_golden.py`)

> Sur la donnée : `tests/fixtures/sample.parquet` est committé (dataset synthétique), reproductible par construction. Sur un vrai dataset on épinglerait l'URL par SHA + un `sha256` committé, voire DVC (`dvc pull` en CI), hors-scope ici.

# 5 - Où vit la qualité du modèle (et le déploiement)

La CI de PR ne juge **pas** la qualité du modèle (récap §1) : l'entraînement lourd et son évaluation tournent dans **KFP**, sur le cluster et le dataset complet. Le hook existe déjà : `src/scripts/quality_gate.py` lit l'AUC d'une version candidate et de `@production` (seuils `[auc]` de `tests/thresholds.toml` : `floor = 0.80`, `max_regression = 0.02`) et refuse si `auc < plancher` ou `auc < prod − régression`. Il est appelé **dans la pipeline KFP** (étape d'éval, avant de poser `@champion`), pas en GHA.

**Et le déploiement ?** GitHub Actions s'arrête aux **gates**. Il ne déploie pas. La livraison est **déclarative** (KServe réconcilie l'`InferenceService`, `k8s/kserve/`) et la promotion `@champion → @production` vit dans KFP (composant `promote`) + le geste humain. Build d'image, `kubectl apply`, rollback, cron de ré-entraînement, durcissement supply-chain : c'est du **DevOps classique** ou du **KFP-natif**, hors de ce module.

# 6 - Rejouer les vraies gates en local avec act

Une fois `ci` et `ml-ci` écrits (§3, §4), on les rejoue en local. Auto-portés, ils ne demandent aucun secret.

Pré-cacher les actions une fois (`.actrc` active `--action-offline-mode`, `act` les réutilise sans re-cloner) :

```bash
git clone --depth 1 -b v4 https://github.com/actions/checkout   ~/.cache/act/actions-checkout@v4
git clone --depth 1 -b v6 https://github.com/astral-sh/setup-uv ~/.cache/act/astral-sh-setup-uv@v6
```

```bash
act pull_request -W .github/workflows/ci.yaml
act pull_request -W .github/workflows/ml-ci.yaml
```


