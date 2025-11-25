# Feelings Analysis - Classification multi-aspects de sentiments

Projet d'analyse de sentiments multi-aspects pour des avis de restaurants en français. Le système classifie automatiquement 4 aspects (**Prix**, **Cuisine**, **Service**, **Ambiance**) selon 4 labels : **Positive**, **Négative**, **Neutre**, **NE** (Non Exprimé).

## Vue d'ensemble

Ce projet implémente deux approches de classification :

1. **LLM** : Classification zero-shot avec des modèles de langage (via Ollama)
2. **PLMFT** : Fine-tuning de modèles pré-entraînés (CamemBERT-Large) avec PyTorch Lightning

**Statut actuel** : Le système PLMFT est pleinement opérationnel avec des optimisations avancées pour atteindre ~85% d'accuracy.

---

## Architecture du Projet

```
FeelingsAnalysis/
├── data/                           # Données d'entraînement et test
│   ├── ftdataset_train.tsv        # Ensemble d'entraînement
│   ├── ftdataset_val.tsv          # Ensemble de validation
│   └── ftdataset_test.tsv         # Ensemble de test
├── src/                            # Code source
│   ├── config.py                  # Configuration et hyperparamètres
│   ├── classifier_wrapper.py      # Wrapper unifié pour LLM/PLMFT
│   ├── llm_classifier.py          # Classificateur LLM zero-shot
│   ├── plm_classifier.py          # Classificateur PLMFT (CamemBERT)
│   ├── runproject.py              # Point d'entrée principal
│   └── lightning_logs/            # Logs d'expériences PyTorch Lightning
│       ├── version_0/
│       ├── version_1/
│       └── version_2/             # Meilleure expérience actuelle
├── requirements.txt               # Dépendances Python
├── install.sh                     # Script d'installation
└── README.md                      # Ce fichier
```

---

## Dataset

**Format** : Fichiers TSV (Tab-Separated Values)

| Colonne    | Description                                      |
|------------|--------------------------------------------------|
| `Avis`     | Texte de l'avis client                          |
| `Prix`     | Sentiment sur le prix                           |
| `Cuisine`  | Sentiment sur la qualité de la cuisine          |
| `Service`  | Sentiment sur le service                        |
| `Ambiance` | Sentiment sur l'ambiance                        |

**Labels possibles** : `Positive`, `Négative`, `Neutre`, `NE` (Non Exprimé)

**Caractéristiques** :
- **Classification multi-sortie** : 4 tâches de classification indépendantes
- **4 classes** par aspect
- Déséquilibre de classes possible (notamment `NE`)

---

## Installation

### Prérequis
- Python 3.8+
- CUDA 11.8+ (pour GPU)
- 16GB+ RAM recommandé

### Installation rapide

```bash
pip install -r requirements.txt
```

### Dépendances principales
- `torch` : Deep Learning
- `transformers` : Modèles pré-entraînés (CamemBERT)
- `lightning` : Framework d'entraînement
- `pandas` : Manipulation de données
- `ollama` : Interface LLM (mode LLM uniquement)

---

## Utilisation

### Configuration de la méthode

Dans [`classifier_wrapper.py`](file:///home/decoopmn/FeelingsAnalysis/src/classifier_wrapper.py#L12), modifier :

```python
METHOD: str = 'PLMFT'  # ou 'LLM'
```

### Lancer l'entraînement et l'évaluation

```bash
cd src/
python runproject.py --device=0 --n_runs=5
```

**Arguments disponibles** :
- `--device=0` : Utiliser GPU 0 (-1 pour CPU)
- `--n_runs=5` : Nombre d'exécutions (moyenne finale)
- `--n_train=1000` : Limiter l'entraînement (défaut: -1 = tout)
- `--n_test=500` : Limiter les tests (défaut: -1 = tout)
- `--batch_size=32` : Taille de batch
- `--learning_rate=2e-5` : Taux d'apprentissage
- `--max_epochs=20` : Nombre d'époques max

---

## Configuration et hyperparamètres

Le fichier [`config.py`](file:///home/decoopmn/FeelingsAnalysis/src/config.py) centralise tous les hyperparamètres.

### Hyperparamètres actuels (optimisés pour version_2)

```python
batch_size: int = 32                    # Taille de batch
accumulate_grad_batches: int = 4        # Gradient accumulation (batch effectif = 128)
learning_rate: float = 2e-5             # LR têtes de classification
backbone_lr: float = 1e-5               # LR backbone CamemBERT (discriminative LR)
max_epochs: int = 20                    # Nombre d'époques max
max_length: int = 256                   # Longueur max des séquences
weight_decay: float = 0.01              # Régularisation L2
warmup_steps: int = 1000                # Steps de warmup (linear)
scheduler: str = "linear"               # Scheduler LR (linear ou cosine)
early_stopping_patience: int = 3        # Early stopping sur val_loss
```

### Optimisations implémentées

**Gradient checkpointing** : Réduit l'utilisation mémoire GPU  
**Mixed precision training** (FP16) : Accélération ~2-3x  
**Discriminative learning rates** : LR différent backbone/têtes  
**Label smoothing** (0.1) : Réduit l'overfitting  
**Gradient accumulation** : Simule de plus gros batchs  
**Warmup + Linear/cosine scheduler** : Stabilise l'entraînement  
**Early stopping** : Arrêt automatique si pas d'amélioration  
**DataLoader optimisé** : `num_workers=8`, `pin_memory=True`, `persistent_workers=True`

---

## Architecture du modèle

### Modèle PLMFT ([`plm_classifier.py`](file:///home/decoopmn/FeelingsAnalysis/src/plm_classifier.py))

```
Input Text
    ↓
[Tokenizer CamemBERT]
    ↓
[CamemBERT-Large Backbone]  ← Gradient Checkpointing activé
    ↓
[CLS Token Pooling]
    ↓
    ├─→ [Linear Layer] → Prix (4 classes)
    ├─→ [Linear Layer] → Cuisine (4 classes)
    ├─→ [Linear Layer] → Service (4 classes)
    └─→ [Linear Layer] → Ambiance (4 classes)
```

**Caractéristiques** :
- **Backbone** : `camembert/camembert-large` (110M paramètres)
- **4 têtes de classification** indépendantes (une par aspect)
- **Loss** : CrossEntropyLoss avec label smoothing (0.1)
- **Optimizer** : AdamW avec discriminative learning rates
- **Scheduler** : Linear warmup + linear/cosine decay

### Classe `PLMClassifier`

**Méthodes principales** :
- `__init__(cfg)` : Initialisation du modèle et tokenizer
- `forward(input_ids, attention_mask)` : Passage avant
- `training_step(batch, batch_idx)` : Step d'entraînement
- `validation_step(batch, batch_idx)` : Step de validation
- `configure_optimizers()` : Configuration optimizer + scheduler
- `predict(text)` : Prédiction sur un texte unique

---

## Résultats et expériences

Les logs d'entraînement sont stockés dans `src/lightning_logs/version_X/`.

### Meilleures performances (version_2)

| Métrique                        | Valeur                                       |
| ------------------------------- | -------------------------------------------- |
| **Val Loss**                    | **1.836**                                    |
| **Val Acc (moyenne 4 classes)** | **0.8599**                                   |
| **Val Acc Ambiance**            | **0.823**                                    |
| **Val Acc Cuisine**             | **0.872**                                    |
| **Val Acc Prix**                | **0.867**                                    |
| **Val Acc Service**             | **0.878**                                    |


### Historique des expériences

Le dossier `lightning_logs/` contient les versions d'expériences avec différents hyperparamètres. Consultez les fichiers `metrics.csv` pour comparer les performances.

---

## 🔧 Modules principaux

### [`runproject.py`](file:///home/decoopmn/FeelingsAnalysis/src/runproject.py)
Script principal d'orchestration :
- Charge les données TSV
- Initialise le ClassifierWrapper
- Lance l'entraînement (si PLMFT)
- Évalue sur le test set
- Calcule les métriques par aspect et macro-accuracy

### [`classifier_wrapper.py`](file:///home/decoopmn/FeelingsAnalysis/src/classifier_wrapper.py)
Wrapper unifié pour gérer LLM et PLMFT :
- `train(train_data, val_data, device)` : Entraînement (PLMFT uniquement)
- `predict(texts, device)` : Prédiction batch

### [`plm_classifier.py`](file:///home/decoopmn/FeelingsAnalysis/src/plm_classifier.py)
Implémentation PyTorch Lightning du modèle multi-aspects :
- Classe `AspectDataset` : Dataset PyTorch pour multi-aspects
- Classe `PLMClassifier` : LightningModule avec CamemBERT

### [`llm_classifier.py`](file:///home/decoopmn/FeelingsAnalysis/src/llm_classifier.py)
Classificateur zero-shot via LLM (Ollama) :
- Génération de prompts structurés
- Parsing des réponses JSON
- Fallback en cas d'erreur

---

## Objectifs et améliorations

### Réalisé
- [x] Fine-tuning CamemBERT-Large multi-aspects
- [x] Optimisations GPU (gradient checkpointing, mixed precision)
- [x] Discriminative learning rates
- [x] Early stopping et schedulers avancés
- [x] Architecture multi-têtes pour 4 aspects
- [x] Évaluation par aspect + macro-accuracy

### En cours
- [ ] Atteindre 93%+ d'accuracy
- [ ] Optimisation des hyperparamètres (grid search)
- [ ] Ensembling de modèles
- [ ] Data augmentation

### Futures améliorations
- [ ] Intégration de modèles plus récents (DeBERTa, RoBERTa-large)
- [ ] Architecture avec attention croisée entre aspects
- [ ] Distillation de modèle pour déploiement
- [ ] Interface web pour démo

---

## Métriques d'évaluation

Le système calcule :
- **Accuracy par aspect** : Prix, Cuisine, Service, Ambiance
- **Macro-accuracy** : Moyenne des 4 aspects
- **Validation loss** : Loss moyenne sur tous les aspects

**Exemple de sortie** :
```
Prix: 87.23%
Cuisine: 89.45%
Service: 85.12%
Ambiance: 83.67%
Macro Accuracy: 86.37%
```

---

## Contributing

Pour améliorer le projet :
1. Analyser les métriques dans `lightning_logs/`
2. Ajuster les hyperparamètres dans `config.py`
3. Lancer de nouvelles expériences
4. Documenter les résultats

---

**Note** : Ce projet a été optimisé à travers 24+ expériences pour maximiser la performance sur la classification multi-aspects de sentiments en français.