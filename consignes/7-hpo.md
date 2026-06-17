# Rechercher automatiquement les meilleurs hyperparamètres

Dans la pipeline d'entraînement du module 3, le modèle est entraîné avec des hyperparamètres **en dur** (`hidden_layer_sizes=(50,)`, `max_iter=100`…). Mais qui dit que ce sont les bons ? **Katib** automatise la recherche : il lance N essais avec des jeux d'hyperparamètres différents et garde le meilleur.


# 1 - Découverte Katib

Avant de brancher Katib sur notre modèle, on le prend en main sur un exemple jouet, pour découvrir l'outil et son UI.

Katib en une phrase : il lance N essais avec des hyperparamètres différents et garde le meilleur. Quatre objets à connaître :

- **`Experiment`** — la campagne : un objectif (métrique + direction), un algorithme, un espace de recherche, un budget (`max_trial_count`, `parallel_trial_count`).
- **`Trial`** — un essai = une exécution avec **un** jeu d'hyperparamètres.
- **`Suggestion`** — propose les prochains hyperparamètres selon l'algorithme (random, bayésien, hyperband…).
- **Metrics Collector** — récupère la métrique de chaque essai. Le collector **StdOut** (par défaut) lit les lignes `<metric>=<valeur>` sur la sortie standard — d'où le `print(f"result={result}")` ci-dessous.

**1. Le script**

Le SDK `kubeflow.katib` : on définit une fonction objectif, un espace de recherche, puis `KatibClient.tune(...)`. On lance le script dans un environnement isolé avec `uv run --no-project --with kubeflow-katib`.

`src/scripts/katib_demo.py` — un exemple jouet qui maximise `F(a, b) = 4a - b²` :

```python
import os

import kubeflow.katib as katib

NAMESPACE = os.environ.get("KUBEFLOW_NAMESPACE", "kubeflow-user-example-com")


def objective(parameters):
    import time
    time.sleep(5)
    result = 4 * int(parameters["a"]) - float(parameters["b"]) ** 2
    # Katib lit la metrique au format <nom>=<valeur>
    print(f"result={result}")


def main():
    parameters = {
        "a": katib.search.int(min=10, max=20),
        "b": katib.search.double(min=0.1, max=0.2),
    }

    name = "tune-experiment"
    client = katib.KatibClient(namespace=NAMESPACE)
    client.tune(
        name=name,
        objective=objective,
        parameters=parameters,
        objective_metric_name="result",
        max_trial_count=12,
        resources_per_trial={"cpu": "2"},
    )
    client.wait_for_experiment_condition(name=name)
    print(client.get_optimal_hyperparameters(name))


if __name__ == "__main__":
    main()
```

**2. Lancer**

```bash
export KUBECONFIG=/tmp/kubeflow-config
export KUBEFLOW_NAMESPACE=kubeflow-user-example-com
uv run --no-project --with kubeflow-katib python src/scripts/katib_demo.py
```

`tune()` crée l'`Experiment`, lance 12 essais (chacun tire un `a`/`b`, exécute `objective`, imprime `result=`), puis renvoie les meilleurs hyperparamètres.

**3. Explorer l'UI**

L'UI Katib fait partie du dashboard Kubeflow. On ouvre un tunnel vers la passerelle Istio (comme au module serving) :

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 9080:80
```

Puis, dans le dashboard, section **Experiments (AutoML)** : on y retrouve `tune-experiment`.

- la **liste des essais** avec leurs hyperparamètres et leur `result`.
- le **meilleur essai** et la courbe de l'objectif au fil des essais.
- la vue qui relie hyperparamètres et métrique (graphe / coordonnées parallèles).


# 2 - Intégrer la HPO dans la pipeline d'entraînement

Place à la pratique. La pipeline d'entraînement du module 3 enchaîne `load_data → train_model → promote_if_better`, avec des hyperparamètres en dur dans `train_model`. **À faire** : insérer un step **Katib** qui choisit les meilleurs hyperparamètres, et faire ré-entraîner `train_model` avec.

**1. Créer le composant `tune_hyperparameters`** (`src/orchestration/components/tuning.py`)


Le composant doit :

- définir une **fonction objectif** qui entraîne **notre modèle churn** (même `MLPClassifier` sklearn, même prétraitement et même split que `train_model`) avec les hyperparamètres reçus, puis imprime `auc=...`.
- décrire l'**espace de recherche** avec `katib.search.int` / `katib.search.double` (sur `hidden_layer_size`, `alpha`, `max_iter`) — **pas** `categorical`, bugué dans la version actuelle du SDK.
- passer l'URL des données à chaque essai via `env_per_trial={"DATA_URL": source_url}` (les Trials Katib n'ont pas accès aux artifacts KFP).
- lancer `tune()` avec `objective_metric_name="auc"`, `objective_type="maximize"`, `algorithm_name="random"`, un budget (`max_trial_count`, `parallel_trial_count`).
- **renvoyer** les meilleurs hyperparamètres : `get_optimal_hyperparameters(name).parameter_assignments` (liste de `{name, value}`) reconditionnés en `NamedTuple`.

**2. Adapter `train_model`** (`src/orchestration/components/training.py`)

Lui ajouter les arguments `hidden_layer_size`, `alpha`, `max_iter` **avec des valeurs par défaut égales à l'ancienne config**. Ainsi la pipeline de *retraining* (module observability), qui appelle `train_model` sans ces arguments, continue de marcher à l'identique.

**3. Brancher dans `training_pipeline`** (`src/orchestration/pipelines/training.py`)

Ajouter un paramètre `namespace`, insérer `tune_hyperparameters` et câbler ses sorties sur `train_model` :

```python
best = tune_hyperparameters(source_url=source_url, namespace=namespace)
trained = train_model(
    dataset=data.outputs["dataset"], mlflow_tracking_uri=mlflow_tracking_uri,
    hidden_layer_size=best.outputs["hidden_layer_size"],
    alpha=best.outputs["alpha"], max_iter=best.outputs["max_iter"],
)
```

**4. Lancer**

```bash
export KUBEFLOW_NAMESPACE=kubeflow-user-example-com
export MLFLOW_TRACKING_URI=https://<votre-mlflow>
uv run python src/orchestration/pipelines/training.py 
```

Puis on peut voir dans la UI ou dans le terminal les trials :

```bash
kubectl get experiments,trials -n $KUBEFLOW_NAMESPACE
```



