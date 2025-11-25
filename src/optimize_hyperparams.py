"""
Hyperparameter Optimization avec Optuna pour FeelingsAnalysis
Optimise automatiquement les hyperparamètres du modèle multi-aspects
"""
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from pathlib import Path
import json

from config import Config
from plm_classifier import PLMClassifier, AspectDataset


def load_data():
    """Charge les données d'entraînement et de validation"""
    df_train = pd.read_csv("../data/ftdataset_train.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    df_val = pd.read_csv("../data/ftdataset_val.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    return df_train.to_dict(orient='records'), df_val.to_dict(orient='records')


def objective(trial: optuna.Trial, device: int = 0, n_epochs: int = 10):
    """
    Fonction objectif pour Optuna
    Retourne la metric à maximiser (val_acc moyenne)
    """
    # Suggérer des hyperparamètres
    cfg = Config()
    
    # Learning rates
    cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    cfg.backbone_lr = trial.suggest_float("backbone_lr", 5e-6, 2e-5, log=True)
    
    # Regularization
    cfg.weight_decay = trial.suggest_float("weight_decay", 0.005, 0.02, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.15)
    
    # Training dynamics
    cfg.batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])
    cfg.max_length = trial.suggest_categorical("max_length", [256, 384, 512])
    cfg.warmup_steps = trial.suggest_int("warmup_steps", 500, 1500, step=250)
    cfg.scheduler = trial.suggest_categorical("scheduler", ["linear", "cosine"])
    
    # Gradient accumulation pour simuler plus gros batch
    cfg.accumulate_grad_batches = trial.suggest_categorical("accumulate_grad_batches", [2, 4, 8])
    
    # Fix max_epochs pour essais rapides
    cfg.max_epochs = n_epochs
    
    # Seed pour reproductibilité
    seed_everything(42 + trial.number)
    
    # Charger données
    train_data, val_data = load_data()
    
    # Initialiser modèle avec label smoothing suggéré
    model = PLMClassifier(cfg)
    # Modifier le label smoothing dans le criterion
    model.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Créer datasets et dataloaders
    train_dataset = AspectDataset(train_data, model.tokenizer, cfg.max_length)
    val_dataset = AspectDataset(val_data, model.tokenizer, cfg.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=False
    )
    
    # Pruning callback pour Optuna
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    
    # Trainer - Force single GPU due to NCCL issues
    # Use GPU 0 only to avoid driver/library version mismatch
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator='gpu',
        devices=[0],  # Single GPU to avoid NCCL issues
        log_every_n_steps=10,
        enable_checkpointing=False,
        callbacks=[early_stop_callback, pruning_callback],
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        precision='16-mixed',
        enable_progress_bar=False,
        enable_model_summary=False
    )
    
    # Entraînement
    try:
        trainer.fit(model, train_loader, val_loader)
        
        # Récupérer la meilleure val_acc moyenne
        # On calcule manuellement la moyenne des 4 aspects
        logged_metrics = trainer.callback_metrics
        
        val_acc_ambiance = logged_metrics.get('val_acc_Ambiance', 0).item()
        val_acc_cuisine = logged_metrics.get('val_acc_Cuisine', 0).item()
        val_acc_prix = logged_metrics.get('val_acc_Prix', 0).item()
        val_acc_service = logged_metrics.get('val_acc_Service', 0).item()
        
        mean_val_acc = (val_acc_ambiance + val_acc_cuisine + val_acc_prix + val_acc_service) / 4.0
        
        # Logger les métriques individuelles
        trial.set_user_attr("val_acc_ambiance", val_acc_ambiance)
        trial.set_user_attr("val_acc_cuisine", val_acc_cuisine)
        trial.set_user_attr("val_acc_prix", val_acc_prix)
        trial.set_user_attr("val_acc_service", val_acc_service)
        
        return mean_val_acc
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--n-epochs", type=int, default=10, help="Max epochs per trial")
    parser.add_argument("--study-name", type=str, default="feelings_optimization", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna DB storage (e.g., sqlite:///optuna.db)")
    args = parser.parse_args()
    
    # Créer ou charger une étude Optuna
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
    
    # Optimisation
    study.optimize(
        lambda trial: objective(trial, device=args.device, n_epochs=args.n_epochs),
        n_trials=args.n_trials,
        timeout=None,
        show_progress_bar=True
    )
    
    # Résultats
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (mean val_acc): {study.best_value:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Métriques best trial (si disponibles)
    if study.best_trial.user_attrs:
        print("\nBest metrics by aspect:")
        if 'val_acc_ambiance' in study.best_trial.user_attrs:
            print(f"  Ambiance: {study.best_trial.user_attrs['val_acc_ambiance']:.4f}")
            print(f"  Cuisine: {study.best_trial.user_attrs['val_acc_cuisine']:.4f}")
            print(f"  Prix: {study.best_trial.user_attrs['val_acc_prix']:.4f}")
            print(f"  Service: {study.best_trial.user_attrs['val_acc_service']:.4f}")
        else:
            print("  (No valid metrics - all trials returned 0.0)")
    
    # Sauvegarder les meilleurs hyperparamètres
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    best_config = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_metrics": {
            "val_acc_ambiance": study.best_trial.user_attrs['val_acc_ambiance'],
            "val_acc_cuisine": study.best_trial.user_attrs['val_acc_cuisine'],
            "val_acc_prix": study.best_trial.user_attrs['val_acc_prix'],
            "val_acc_service": study.best_trial.user_attrs['val_acc_service']
        }
    }
    
    with open(output_dir / "best_hyperparams.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest hyperparameters saved to {output_dir / 'best_hyperparams.json'}")
    
    # Sauvegarder tous les trials
    df_trials = study.trials_dataframe()
    df_trials.to_csv(output_dir / "all_trials.csv", index=False)
    print(f"All trials saved to {output_dir / 'all_trials.csv'}")
    
    # Visualisations (si plotly disponible)
    try:
        import plotly
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate
        )
        
        fig1 = plot_optimization_history(study)
        fig1.write_html(output_dir / "optimization_history.html")
        
        fig2 = plot_param_importances(study)
        fig2.write_html(output_dir / "param_importances.html")
        
        fig3 = plot_parallel_coordinate(study)
        fig3.write_html(output_dir / "parallel_coordinate.html")
        
        print(f"\nVisualizations saved to {output_dir}/")
    except ImportError:
        print("\nPlotly not available, skipping visualizations")


if __name__ == "__main__":
    main()
