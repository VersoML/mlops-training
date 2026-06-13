# Introduction — du modèle entraîné au endpoint de prédiction

À la fin du module repro, on a un modèle versionné dans le registry MLflow. Mais un modèle dans un registry ne sert personne : il faut le rendre exploitable pour produire des prédictions. C'est le *serving*.

Servir un modèle peut se faire de deux manières (principalement) :

En **batch offline**, on charge le modèle, on le fait tourner sur un gros lot de données et on stocke les prédictions (typiquement un job planifié). C'est une option très intéressante, souvent la plus simple et la moins coûteuse quand les prédictions n'ont pas besoin d'être instantanées.

Ici on se concentre sur l'**online serving** : exposer le modèle derrière une API qui reçoit des features et renvoie une prédiction à la demande, en temps réel.

Il n'y a pas une seule bonne façon de servir un modèle en ligne : le bon choix dépend du contexte (smoke-test local, prototype, prod sur Kubernetes). On va donc parcourir les options de la plus simple à la plus complète, chacune levant une limite de la précédente :

1. **MLflow serve** — une commande, un endpoint. Idéal pour un smoke-test, mais routes et schémas imposés.
2. **FastAPI** — sa propre API : validation typée (Pydantic), routes métier, doc OpenAPI.
3. **Docker** — empaqueter le serveur dans une image reproductible, déployable n'importe où.
4. **Kubernetes (vanilla)** — déployer l'image sur un cluster : Deployment, Service, probes.
5. **KServe** — une surcouche Kubernetes dédiée au serving de modèles : autoscaling, runtimes natifs, protocoles d'inférence standard. On l'explore sous plusieurs angles (image MLflow autonome, runtime natif, predictor custom, déploiement par SDK).


# 1 - serving with mlflow

MLflow sait servir un modèle du registry directement, sans écrire une ligne de code : `mlflow models serve` charge le modèle et lance un serveur HTTP avec des routes prêtes à l'emploi (`/invocations`, `/ping`). C'est le moyen le plus rapide de vérifier qu'un modèle répond parfait pour un smoke-test, avant de construire une vraie API.

**1. Servir directement avec MLflow**

```bash
uv run mlflow models serve --model-uri models:/churn-model@champion --port 8000 --no-conda
```



**2. Tester le serveur MLflow**

```bash
curl -w '\n' http://127.0.0.1:8000/ping
```

```bash
curl -X POST http://127.0.0.1:8000/invocations \
  -w '\n' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_records": [{
      "tenure_days": 200,
      "nb_logins_30j": 12,
      "nb_features_used": 4,
      "plan": "Pro",
      "support_tickets_90j": 1,
      "mrr_eur": 49.0,
      "has_integration": true
    }]
  }'
```

**3. Ce qu'on n'a pas**

`mlflow models serve` est parfait pour un smoke-test, mais :

- Pas de **validation typée** des inputs (`tenure_days = "abc"` peut passer jusqu'au modèle)
- **Routes imposées** (`/invocations`, `/ping`, `/version`), pas de routes métier
- **Schémas de réponse** non typés (le frontend doit deviner)
- Pas de **middleware** (auth, rate-limit, logging custom)
- **Protocole MLflow** imposé (`dataframe_records`, `dataframe_split`) — pas un contrat REST classique
  
# 2 - FastAPI

Pour une vraie API, on écrit notre propre serveur avec FastAPI : un framework web Python qui valide les entrées et sorties à partir de types Pydantic et génère automatiquement la doc OpenAPI. On gagne ce que `mlflow models serve` n'offre pas : validation typée, routes métier (`/predict`, `/health`), schémas de réponse explicites.

**1. Créer `src/server.py`**

```python
import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")


class ChurnFeatures(BaseModel):
    tenure_days: int = Field(ge=0)
    nb_logins_30j: int = Field(ge=0)
    nb_features_used: int = Field(ge=0)
    plan: Literal["Free", "Pro", "Enterprise"]
    support_tickets_90j: int = Field(ge=0)
    mrr_eur: float = Field(ge=0)
    has_integration: bool


class Prediction(BaseModel):
    churned: bool


class BatchRequest(BaseModel):
    instances: list[ChurnFeatures]


class BatchResponse(BaseModel):
    predictions: list[Prediction]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if "model" in state else "loading"}


@app.post("/predict", response_model=Prediction)
def predict(features: ChurnFeatures) -> Prediction:
    try:
        df = pd.DataFrame([features.model_dump()])
        prediction = state["model"].predict(df)
        return Prediction(churned=bool(prediction[0]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    if not request.instances:
        raise HTTPException(status_code=400, detail="instances must not be empty")
    try:
        df = pd.DataFrame([instance.model_dump() for instance in request.instances])
        predictions = state["model"].predict(df)
        return BatchResponse(predictions=[Prediction(churned=bool(p)) for p in predictions])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
```

Le `lifespan` charge le modèle une fois au démarrage, pas à chaque requête.

**2. Lancer le serveur custom**

Arrêter d'abord le serveur MLflow puis :

```bash
export MODEL_URI=models:/churn-model@champion 
uv run uvicorn src.serving.server:app --reload --port 8000
```

**3. Tester**

```bash
curl -w '\n' http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -w '\n' \
  -H 'Content-Type: application/json' \
  -d '{
    "tenure_days": 200,
    "nb_logins_30j": 12,
    "nb_features_used": 4,
    "plan": "Pro",
    "support_tickets_90j": 1,
    "mrr_eur": 49.0,
    "has_integration": true
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -w '\n' \
  -H 'Content-Type: application/json' \
  -d '{"instances": [{"tenure_days": 30, "nb_logins_30j": 1, "nb_features_used": 1, "plan": "Free", "support_tickets_90j": 5, "mrr_eur": 0.0, "has_integration": false}]}'
```

# 3 - Conteneurisation Docker

Le serveur FastAPI tourne pour l'instant sur notre machine, avec nos dépendances locales. Pour le déployer ailleurs (cluster, cloud, etc.) sans se soucier de l'environnement, on l'empaquette dans une image Docker : le code, ses dépendances figées (`uv.lock`) et la commande de lancement, le tout reproductible et déployable n'importe où.

**1. Créer le `Dockerfile.serve`**

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn serving.server:app --host 0.0.0.0 --port ${PORT}"]
```

**2. Builder l'image**

```bash
docker build -f Dockerfile.serve -t churn-api:latest .
```

**3. Lancer le container**

Le tracking server est public, le container y accède directement via Internet :

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=https://<votre-mlflow> \
  -e MODEL_URI=models:/churn-model@champion \
  churn-api:latest
```

**4. Tester depuis l'hôte**

Mêmes commandes `curl` que Partie 1 depuis l'hôte, vers `127.0.0.1:8000`.


Remarque : 

Pour un déploiement "ferme" (image autonome, pas de tracking server à joindre), on peut **embarquer le modèle dans l'image** au moment du build. Pratique pour la prod où on veut l'image indépendante du registry. Trade-off : il faut rebuild à chaque promotion.

# 4 - Kubernetes

Une image, c'est bien, mais en lancer un seul conteneur à la main ne tient pas en prod : il faut redémarrer le service s'il crashe, scaler à plusieurs instances, router le trafic, faire des mises à jour sans coupure. Une image Docker se déploie d'ailleurs telle quelle sur un service managé comme Google Cloud Run ou AWS App Runner, qui gèrent ce scaling et ce routing pour nous. C'est souvent le choix le plus pragmatique. Mais ces services restent des boîtes plus fermées : dès qu'on veut un contrôle fin (placement, sidecars, autoscaling custom, intégration avec un écosystème ML comme Kubeflow/KServe), on se tourne vers un orchestrateur complet, Kubernetes.

On déploie ici notre image sur un cluster Kind (Kubernetes local) avec les briques de base : un *Deployment* qui maintient les pods en vie, un *Service* qui leur donne une adresse stable, et des *probes* qui surveillent leur santé.

### Prérequis : installer le cluster Kubeflow

Toutes les parties qui suivent (K8s vanilla ici, puis KServe aux parties 5-9) tournent sur le même cluster local, nommé `kubeflow`. On l'installe une fois via le repo officiel `[kubeflow/manifests](https://github.com/kubeflow/manifests)`, qui fournit à la fois le **script de création du cluster** (KinD + `kubectl` + `kustomize` aux versions testées) et l'**overlay `example`** (tous les composants Kubeflow, dont Knative et Istio dont dépend KServe). Toutes les commandes ci-dessous se lancent depuis le repo cloné.

```bash
git clone https://github.com/kubeflow/manifests.git
cd manifests

./tests/install_KinD_create_KinD_cluster_install_kustomize.sh
```

Pointer `kubectl` sur le cluster :

```bash
kind get kubeconfig --name kubeflow > /tmp/kubeflow-config
export KUBECONFIG=/tmp/kubeflow-config
```

Appliquer tous les composants Kubeflow (la boucle réessaie tant que des CRDs ne sont pas encore prêtes) :

```bash
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 20; done
```

Attendre que tous les namespaces soient `Running` :

```bash
kubectl get pods -n cert-manager
kubectl get pods -n istio-system
kubectl get pods -n auth
kubectl get pods -n oauth2-proxy
kubectl get pods -n knative-serving
kubectl get pods -n kubeflow
kubectl get pods -n kubeflow-user-example-com
```

**1. Charger l'image dans Kind**

On pousser explicitement l'image docker dans kind :

```bash
kind load docker-image churn-api:latest --name kubeflow
```

**2. Écrire les manifestes**

`k8s/configmap.yaml` — un *ConfigMap* stocke des paires clé/valeur que K8s injecte ensuite comme variables d'environnement dans les pods :

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: churn-api-config
data:
  MLFLOW_TRACKING_URI: "https://<votre-mlflow>"
  MODEL_URI: "models:/churn-model@champion"
```

Le tracking server étant public, les pods le joignent directement via Internet.

`k8s/deployment.yaml` — un *Deployment* décrit l'état voulu de l'app (image, nombre de replicas, probes, ressources) et K8s la maintient en vie :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: churn-api
  template:
    metadata:
      labels:
        app: churn-api
    spec:
      containers:
        - name: api
          image: churn-api:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: churn-api-config
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            failureThreshold: 60
            periodSeconds: 5
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 10
            failureThreshold: 3
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
```

`k8s/service.yaml` — un *Service* donne aux pods une adresse interne stable et répartit le trafic entre eux (les pods sont éphémères, leur IP change) :

```yaml
apiVersion: v1
kind: Service
metadata:
  name: churn-api
spec:
  selector:
    app: churn-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

**3. Appliquer**

```bash
kubectl apply -f k8s/raw/
kubectl get pods -w
```

**4. Tester**

Ouvrir un tunnel vers le Service :

```bash
kubectl port-forward svc/churn-api 8080:80
```

Dans un autre terminal, interroger l'API (le Service écoute sur le port 80, redirigé en local sur 8080) :

```bash
curl -w '\n' http://127.0.0.1:8080/health
```

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -w '\n' \
  -H 'Content-Type: application/json' \
  -d '{
    "tenure_days": 200,
    "nb_logins_30j": 12,
    "nb_features_used": 4,
    "plan": "Pro",
    "support_tickets_90j": 1,
    "mrr_eur": 49.0,
    "has_integration": true
  }'
```

**5. Pourquoi un `startupProbe` ?**

Le chargement du modèle au démarrage peut prendre plusieurs dizaines de secondes. Sans `startupProbe`, la `livenessProbe` se déclenche pendant ce temps, échoue, et K8s tue le pod en boucle (`CrashLoopBackOff`). Le `startupProbe` suspend les autres probes le temps du démarrage.
Une règle générale : dès qu'un container fait un init non trivial (chargement modèle, warm-up, migration DB), il en faut un.

**6. Est-ce que c'est suffisant ?**

Ce qu'on vient de faire (Deployment + Service + probes + limites de ressources) est le **minimum** pour faire tourner un modèle sur Kubernetes : le pod redémarre s'il crashe, le Service lui donne une adresse stable, les probes évitent de router vers un pod pas prêt. C'est suffisant pour servir, pas pour de la prod sérieuse.

Il manque encore beaucoup, et chaque manque se fait à la main en K8s vanilla :

- **Autoscaling** — un seul replica fixe : il faudrait un HPA (scaling sur CPU/RPS), voire le scale-to-zero quand le modèle n'est pas sollicité.
- **Déploiements progressifs** — pas d'A/B ni de canary : on ne peut pas router 10 % du trafic vers une nouvelle version pour la valider sans risque.
- **Observabilité** — pas de métriques de latence/erreur/throughput exposées automatiquement (Prometheus/Grafana).
- **Sécurité réseau** — pas de TLS, d'auth, ni de gestion propre des secrets.
- **Spécificités ML** — versionnage des modèles déployés, protocoles d'inférence standard, choix d'un runtime selon le format du modèle.

Deux choix s'offrent à nous : tout recâbler nous-mêmes par-dessus Kubernetes, ou adopter une surcouche dédiée au serving de modèles qui apporte ces fonctionnalités. 

**KServe** est exactement ça : construit sur Kubernetes (et Knative), il gère autoscaling et scale-to-zero, canary, runtimes ML natifs et protocoles d'inférence standard.

# 5 - KServe : MLflow Integration


Sur KServe, on déploie un modèle via un `InferenceService` : la ressource Kubernetes qui décrit *quoi servir* et *comment*, et délègue à KServe tout ce qu'on devait câbler à la main en vanilla (autoscaling, routing, scale-to-zero, etc.).

Reste à dire à l'`InferenceService` quoi servir. La voie la plus directe quand on part d'un modèle MLflow : laisser MLflow construire lui-même une image Docker autonome qui embarque le modèle et son serveur d'inférence (`mlflow models build-docker`), puis pointer l'`InferenceService` dessus. C'est l'approche officielle documentée par MLflow ([docs](https://mlflow.org/docs/latest/ml/deployment/deploy-model-to-kubernetes/tutorial/)).



**1. Configurer la cible**

On déploie dans le namespace de l'utilisateur Kubeflow (`kubeflow-user-example-com`) pour que les modèles apparaissent dans la UI Models web app :

```bash
kubectl config set-context --current --namespace=kubeflow-user-example-com
export MLFLOW_TRACKING_URI=https://<votre-mlflow>
```

**2. Builder l'image Docker du modèle**

```bash
uv run mlflow models build-docker \
  --model-uri models:/churn-model@champion \
  --name churn-mlflow:latest
```

En une commande, MLflow produit une image autonome prête à servir :

- télécharge l'artefact du modèle (`@champion`) depuis le registry.
- lit le `requirements.txt` loggé avec le modèle pour reconstruire son environnement exact (mêmes versions qu'à l'entraînement).
- génère un Dockerfile qui embarque le modèle, ses dépendances et un serveur d'inférence (FastAPI, exposant `/invocations` et `/ping`).
- build l'image en local sous le nom `churn-mlflow:latest`.

L'image résultante est indépendante du registry : elle contient le modèle, on peut la déployer sans tracking server à joindre.


**3. Charger l'image dans Kind (mode Serverless / Knative)**

L'image vient d'être buildée en local : il faut la rendre disponible au cluster Kind, qui ne voit pas le démon Docker de l'hôte.

```bash
docker tag churn-mlflow:latest dev.local/churn-mlflow:latest
kind load docker-image dev.local/churn-mlflow:latest --name kubeflow
```

On retague sous `dev.local/` parce que KServe tourne en mode **Serverless** (Knative) : sans ce préfixe, Knative tente de résoudre l'image dans un registry distant et la Revision ne démarre pas. `dev.local/` est justement un préfixe que Knative laisse résoudre localement.

**4. Écrire l'`InferenceService`**

`kserve/inference-service-mlflow.yaml` — un *InferenceService* est la ressource KServe qui déploie et expose un modèle. Ici elle lance simplement l'image autonome :

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-mlflow
  namespace: kubeflow-user-example-com
spec:
  predictor:
    minReplicas: 1
    containers:
      - name: kserve-container
        image: dev.local/churn-mlflow:latest
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8000
            protocol: TCP
```

**5. Appliquer**

```bash
kubectl apply -f k8s/kserve/inference-service-mlflow.yaml
kubectl get revisions -l serving.kserve.io/inferenceservice=churn-mlflow -w
kubectl wait --for=condition=Ready inferenceservice/churn-mlflow --timeout=300s
```


**6. Tester via port-forward**

Port-forward le pod du predictor par label (port 8000 = serveur mlflow) :

```bash
kubectl port-forward "$(kubectl get pod -l serving.kserve.io/inferenceservice=churn-mlflow -o name | head -1)" 8888:8000
```

Dans un autre terminal :

```bash
curl -X POST http://localhost:8888/invocations \
  -w '\n' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_records": [{
      "tenure_days": 200,
      "nb_logins_30j": 12,
      "nb_features_used": 4,
      "plan": "Pro",
      "support_tickets_90j": 1,
      "mrr_eur": 49.0,
      "has_integration": true
    }]
  }'
```

Réponse attendue : `{"predictions": [false]}` (ou `true` selon les features).

L'image utilise le **protocole MLflow** (`/invocations`, `dataframe_records`), pas le protocole v2 de KServe. C'est ce qu'attend l'image générée par `build-docker`.



# 6 - KServe : InferenceService manuel + sélection du runtime

À la partie 5, on a fait servir le modèle par une image qu'on a buildée nous-mêmes. Mais pour les formats de modèles courants (sklearn, XGBoost, PyTorch, Tensorflow…), KServe sait servir un modèle **sans aucune image à construire** : on lui donne juste l'emplacement du modèle (`storageUri`) et le format (`modelFormat`), et il choisit tout seul un *ServingRuntime* (un serveur d'inférence préinstallé sur le cluster, capable de charger ce format). Plus léger que l'image autonome, et c'est l'usage le plus standard de KServe.

Un *ServingRuntime* est donc une image de serveur générique (ex. `kserve-sklearnserver`) qui sait charger un modèle d'un format donné et l'exposer via un protocole d'inférence standard. Plusieurs runtimes peuvent supporter le même format.



**1. Auto-sélection du runtime**

On déclare un `InferenceService` minimal : pas d'image, juste le format et l'emplacement du modèle. 

Ici le `storageUri` pointe vers un modèle sklearn déjà entraîné et publié sur un dépôt distant (`https://github.com/VersoML/mlops-training-data.git`), pas besoin de réentraîner ni de passer par notre registry MLflow, KServe va le récupérer directement depuis cette URL.

```yaml
# kserve/inference-service.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-auto
  namespace: kubeflow-user-example-com
spec:
  predictor:
    minReplicas: 1
    model:
      modelFormat:
        name: sklearn
      storageUri: https://github.com/VersoML/mlops-training-data.git
```

Pas de runtime explicite. KServe regarde tous les `ServingRuntime` présents, filtre ceux qui supportent `sklearn`, et sélectionne celui de plus haute priorité.

```bash
kubectl apply -f k8s/kserve/inference-service.yaml
kubectl wait --for=condition=Ready inferenceservice/churn-auto --timeout=300s
kubectl get inferenceservice churn-auto -o yaml | grep -A2 runtime
```

`grep` montre le runtime choisi (ici `kserve-sklearnserver`). Pour le tester localement, on ouvre un tunnel vers le pod : en mode Serverless, l'URL publique (`*.example.com`) ne résout pas sur notre machine, donc on `port-forward` directement le pod du predictor, ciblé par label. Le runtime parle le **protocole v2** (Open Inference) sur le port **8080** :

```bash
kubectl port-forward "$(kubectl get pod -l serving.kserve.io/inferenceservice=churn-auto -o name | head -1)" 8888:8080
```

Dans un autre terminal (`content_type: "pd"` assemble les inputs en colonnes nommées, `plan` en `str`) :

```bash
curl -X POST http://localhost:8888/v2/models/churn-auto/infer \
  -w '\n' \
  -H 'Content-Type: application/json' \
  -d '{
    "parameters": {"content_type": "pd"},
    "inputs": [
      {"name": "tenure_days",         "shape": [1], "datatype": "INT64", "data": [200]},
      {"name": "nb_logins_30j",       "shape": [1], "datatype": "INT64", "data": [12]},
      {"name": "nb_features_used",    "shape": [1], "datatype": "INT64", "data": [4]},
      {"name": "plan",                "shape": [1], "datatype": "BYTES", "parameters": {"content_type": "str"}, "data": ["Pro"]},
      {"name": "support_tickets_90j", "shape": [1], "datatype": "INT64", "data": [1]},
      {"name": "mrr_eur",             "shape": [1], "datatype": "FP64", "data": [49.0]},
      {"name": "has_integration",     "shape": [1], "datatype": "BOOL", "data": [true]}
    ]
  }'
```

Réponse attendue : `{"outputs":[{... "data":[false]}]}` (ou `true`).

**2. Spécifier explicitement le runtime**

L'auto-sélection est pratique mais implicite : si la priorité d'un autre runtime sklearn change, ou qu'un nouveau runtime est installé, KServe pourrait en choisir un autre sans qu'on s'en rende compte. Pour un déploiement reproductible, on nomme le runtime voulu via le champ `runtime`. KServe ne choisit plus, il utilise celui qu'on lui impose.

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-explicit
  namespace: kubeflow-user-example-com
spec:
  predictor:
    minReplicas: 1
    model:
      modelFormat:
        name: sklearn
      runtime: kserve-sklearnserver
      storageUri: https://github.com/VersoML/mlops-training-data.git
```

```bash
kubectl apply -f k8s/kserve/inference-service-explicit.yaml
kubectl wait --for=condition=Ready inferenceservice/churn-explicit --timeout=300s
```

On **force** le runtime par son nom au lieu de laisser KServe choisir. On le teste comme `churn-auto` :

```bash
kubectl port-forward "$(kubectl get pod -l serving.kserve.io/inferenceservice=churn-explicit -o name | head -1)" 8888:8080
# autre terminal :
curl -X POST http://localhost:8888/v2/models/churn-explicit/infer \
  -w '\n' -H 'Content-Type: application/json' \
  -d '{"parameters":{"content_type":"pd"},"inputs":[{"name":"tenure_days","shape":[1],"datatype":"INT64","data":[200]},{"name":"nb_logins_30j","shape":[1],"datatype":"INT64","data":[12]},{"name":"nb_features_used","shape":[1],"datatype":"INT64","data":[4]},{"name":"plan","shape":[1],"datatype":"BYTES","parameters":{"content_type":"str"},"data":["Pro"]},{"name":"support_tickets_90j","shape":[1],"datatype":"INT64","data":[1]},{"name":"mrr_eur","shape":[1],"datatype":"FP64","data":[49.0]},{"name":"has_integration","shape":[1],"datatype":"BOOL","data":[true]}]}'
```

> **Attention au runtime qu'on fige.** Un runtime natif embarque ses propres versions de librairies. On fige `kserve-sklearnserver` parce qu'il charge bien notre modèle. On fige donc un runtime qu'on a **validé** avec son modèle.

**3. Comparer**

```bash
kubectl get inferenceservice
kubectl describe inferenceservice churn-explicit
```

Vérifier les pods générés et le runtime effectif des deux `InferenceService`.

**4. auto vs explicit**


| Auto-sélection                                       | Explicit                        |
| ---------------------------------------------------- | ------------------------------- |
| Plus simple à écrire                                 | Plus reproductible              |
| Peut changer si la priorité d'un autre runtime monte | Verrouillé sur un runtime nommé |
| OK pour prototypage                                  | Recommandé en prod              |


Pour la prod : toujours expliciter, et fixer une version d'image dans le `ServingRuntime` lui-même.



# 7 - KServe : Custom predictor

Les runtimes natifs (partie 6) chargent un modèle d'un format standard et le servent tel quel : entrée → `model.predict()` → sortie. Mais dès qu'il faut du code autour de la prédiction (pré-traitement métier, enrichissement des features, post-processing, agrégation de plusieurs modèles, ... ) un runtime générique ne suffit plus.

KServe permet alors d'écrire son **propre predictor** : une classe Python qui hérite de `kserve.Model` et implémente `load()` (charger le modèle au démarrage) et `predict()` (transformer le payload, prédire, mettre en forme la réponse). On a le contrôle total du code servi, au prix de devoir builder et maintenir sa propre image.

On reprend ici le modèle de notre registry MLflow (`models:/churn-model@champion`), chargé dans `load()`, et on déploie l'image custom via un `InferenceService` qui pointe dessus.


**1. Écrire `src/serving/ml_server.py`**

```python
import os
from typing import Dict

import mlflow
import pandas as pd
from kserve import Model, ModelServer


MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")


class ChurnPredictor(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None

    def load(self) -> None:
        self.model = mlflow.pyfunc.load_model(MODEL_URI)
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        instances = payload["instances"]
        df = pd.DataFrame(instances)
        predictions = self.model.predict(df)
        return {"predictions": [bool(p) for p in predictions]}


if __name__ == "__main__":
    model = ChurnPredictor("churn-custom")
    model.load()
    ModelServer().start([model])
```

**2. Créer le `Dockerfile.predict`**

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src
EXPOSE 8080

CMD ["uv", "run", "python", "src/serving/ml_server.py"]
```


**3. Build et load dans Kind**

```bash
docker build -f Dockerfile.predict -t churn-custom:latest .
docker tag churn-custom:latest dev.local/churn-custom:latest
kind load docker-image dev.local/churn-custom:latest --name kubeflow
```

**4. YAML `InferenceService` avec image custom**

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: churn-custom
  namespace: kubeflow-user-example-com
spec:
  predictor:
    minReplicas: 1
    containers:
      - name: kserve-container
        image: dev.local/churn-custom:latest
        imagePullPolicy: IfNotPresent
        env:
          - name: MLFLOW_TRACKING_URI
            value: "https://<votre-mlflow>"
        ports:
          - containerPort: 8080
            protocol: TCP
```


```bash
kubectl apply -f k8s/kserve/inference-service-custom.yaml
kubectl wait --for=condition=Ready inferenceservice/churn-custom --timeout=300s
```

**5. Tester via port-forward**

```bash
kubectl port-forward "$(kubectl get pod -l serving.kserve.io/inferenceservice=churn-custom -o name | head -1)" 8888:8080
```

Le predictor custom expose le **protocole v1** de KServe (`kserve.Model.predict` reçoit le payload tel quel, le format est libre). Dans un autre terminal :

```bash
curl -X POST http://localhost:8888/v1/models/churn-custom:predict \
  -w '\n' \
  -H 'Content-Type: application/json' \
  -d '{"instances": [{"tenure_days": 200, "nb_logins_30j": 12, "nb_features_used": 4, "plan": "Pro", "support_tickets_90j": 1, "mrr_eur": 49.0, "has_integration": true}]}'
```


# 8 - Déploiement via le KServe SDK

Jusqu'ici on déploie par `kubectl apply` d'un YAML. Le SDK Python KServe fait la même chose depuis du code. Très pratique pour automatiser depuis une pipeline déjà en Python par exemple. 


Dans `src/scripts/deploy.py` construire l'`InferenceService` et le crée (ou le patche s'il existe déjà) :

```python
isvc = V1beta1InferenceService(
    api_version=constants.KSERVE_V1BETA1,
    kind=constants.KSERVE_KIND_INFERENCESERVICE,
    metadata=client.V1ObjectMeta(name=name, namespace=namespace),
    spec=V1beta1InferenceServiceSpec(
        predictor=V1beta1PredictorSpec(
            min_replicas=min_replicas,
            containers=[client.V1Container(
                name="kserve-container",
                image=image,
                image_pull_policy="IfNotPresent",
                ports=[client.V1ContainerPort(container_port=port, protocol="TCP")],
                env=[client.V1EnvVar(name="MLFLOW_TRACKING_URI", value=tracking_uri)],
            )],
        )
    ),
)
kclient = KServeClient()
kclient.create(isvc, namespace=namespace)   # 409 (existe déjà) -> kclient.patch(...), idempotent
kclient.wait_isvc_ready(name, namespace=namespace)
```

On déploie sous un nouveau nom `churn-sdk` :

```bash
uv run python src/scripts/deploy.py --name churn-sdk
kubectl get inferenceservice churn-sdk
```

On le teste comme `churn-custom`. Le nom de modèle dans l'URL v1 reste `churn-custom`, il vient de `ChurnPredictor("churn-custom")` dans `ml_server.py`, pas du nom de l'`InferenceService` :

```bash
kubectl port-forward "$(kubectl get pod -l serving.kserve.io/inferenceservice=churn-sdk -o name | head -1)" 8888:8080
```

```bash
curl -X POST http://localhost:8888/v1/models/churn-custom:predict \
  -w '\n' -H 'Content-Type: application/json' \
  -d '{"instances": [{"tenure_days": 200, "nb_logins_30j": 12, "nb_features_used": 4, "plan": "Pro", "support_tickets_90j": 1, "mrr_eur": 49.0, "has_integration": true}]}'
```


# 9 - Explorer les modèles dans le dashboard Kubeflow

Jusqu'ici on a tout inspecté en ligne de commande (`kubectl get/describe inferenceservice`). C'est précis mais peu pratique pour avoir une vue d'ensemble. Le cluster Kubeflow embarque la **Models web app** de KServe : une UI qui liste tous les `InferenceService` (statut, runtime, URL, révisions) et permet de cliquer sur chacun pour voir son détail. C'est l'équivalent graphique des commandes précédentes, utile pour visualiser d'un coup d'œil les modèles déployés aux parties 5-8.

**1. Ouvrir le dashboard Kubeflow**

Le dashboard Kubeflow est exposé par l'ingress Istio du cluster. On ouvre un tunnel local vers cette passerelle :

```bash
kubectl port-forward svc/istio-ingressgateway -n istio-system 9080:80
```

Ouvrir http://localhost:9080 et se connecter via Dex (le système d'authentification de Kubeflow) avec les identifiants par défaut `user@example.com` / `12341234`.

**2. Naviguer vers les modèles**

Dans la barre latérale, ouvrir **Endpoints** (la Models web app KServe). On y retrouve `churn-auto`, `churn-explicit`, `churn-mlflow`, `churn-custom`, `churn-sdk`… chacun avec son statut Ready, son runtime/predictor, ses révisions et son URL interne.

**3. Inspecter un modèle**

Cliquer sur un `InferenceService` ouvre le détail : conditions, révisions, YAML, logs, et  si l'observabilité est branchée, les métriques de requêtes. C'est l'équivalent graphique des `kubectl get/describe inferenceservice` des parties précédentes.

