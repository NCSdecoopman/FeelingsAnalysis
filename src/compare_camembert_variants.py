"""
Script pour comparer CamemBERT-base vs CamemBERT-large
Entraîne les deux modèles avec les mêmes hyperparamètres optimisés
"""
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader
import pandas as pd
import json
from pathlib import Path
import argparse
from dataclasses import dataclass

from config import Config
from plm_classifier import AspectDataset
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from lightning import LightningModule


@dataclass
class ModelConfig:
    """Configuration pour les variantes de modèles"""
    model_name: str
    batch_size: int = 32
    learning_rate: float = 2e-5
    backbone_lr: float = 1e-5
    max_epochs: int = 15
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    scheduler: str = "linear"
    early_stopping_patience: int = 3
    accumulate_grad_batches: int = 4
    device: int = 0


class PLMClassifierVariant(LightningModule):
    """Version du classifier qui accepte un model_name paramétrable"""
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.aspects = ['Prix', 'Cuisine', 'Service', 'Ambiance']
        self.label_map = {'Positive': 0, 'Négative': 1, 'Neutre': 2, 'NE': 3}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            add_pooling_layer=False
        )
        self.backbone.gradient_checkpointing_enable()
        
        # Dégeler le backbone
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # 4 têtes de classification
        hidden_size = self.backbone.config.hidden_size
        self.classifiers = nn.ModuleDict({
            aspect: nn.Linear(hidden_size, 4)
            for aspect in self.aspects
        })
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits = {
            aspect: self.classifiers[aspect](cls_output)
            for aspect in self.aspects
        }
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        
        total_loss = 0
        for i, aspect in enumerate(self.aspects):
            loss = self.criterion(logits[aspect], labels[:, i])
            total_loss += loss
            self.log(f'train_loss_{aspect}', loss, prog_bar=False)
        
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        
        total_loss = 0
        for i, aspect in enumerate(self.aspects):
            loss = self.criterion(logits[aspect], labels[:, i])
            total_loss += loss
            
            preds = torch.argmax(logits[aspect], dim=1)
            acc = (preds == labels[:, i]).float().mean()
            self.log(f'val_acc_{aspect}', acc, prog_bar=False)
        
        self.log('val_loss', total_loss, prog_bar=True)
        return total_loss
    
    def configure_optimizers(self):
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.cfg.backbone_lr},
            {'params': classifier_params, 'lr': self.cfg.learning_rate}
        ], weight_decay=self.cfg.weight_decay)
        
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            if current_step < self.cfg.warmup_steps:
                return float(current_step) / float(max(1, self.cfg.warmup_steps))
            else:
                total_steps = max(1, self.trainer.estimated_stepping_batches)
                progress = (current_step - self.cfg.warmup_steps) / float(total_steps - self.cfg.warmup_steps)
                if self.cfg.scheduler == "cosine":
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                else:
                    return max(0.0, 1.0 - progress)
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def load_data():
    df_train = pd.read_csv("../data/ftdataset_train.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    df_val = pd.read_csv("../data/ftdataset_val.tsv", sep=' *\t *', encoding='utf-8', engine='python')
    return df_train.to_dict(orient='records'), df_val.to_dict(orient='records')


def train_model(model_name: str, cfg: ModelConfig, train_data, val_data):
    """Entraîne un modèle avec la configuration donnée"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}\n")
    
    seed_everything(42)
    
    # Créer le modèle
    model_cfg = ModelConfig(
        model_name=model_name,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        backbone_lr=cfg.backbone_lr,
        max_epochs=cfg.max_epochs,
        max_length=cfg.max_length,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        scheduler=cfg.scheduler,
        early_stopping_patience=cfg.early_stopping_patience,
        accumulate_grad_batches=cfg.accumulate_grad_batches
    )
    
    model = PLMClassifierVariant(model_cfg)
    
    # Datasets
    train_dataset = AspectDataset(train_data, model.tokenizer, model_cfg.max_length)
    val_dataset = AspectDataset(val_data, model.tokenizer, model_cfg.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_cfg.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=model_cfg.early_stopping_patience,
        mode='min',
        verbose=True
    )
    
    # Trainer - Force single GPU
    trainer = Trainer(
        max_epochs=model_cfg.max_epochs,
        accelerator='gpu',
        devices=[0],  # Single GPU to avoid NCCL issues
        log_every_n_steps=10,
        enable_checkpointing=False,
        callbacks=[early_stop_callback],
        accumulate_grad_batches=model_cfg.accumulate_grad_batches,
        precision='16-mixed'
    )
    
    # Entraînement
    trainer.fit(model, train_loader, val_loader)
    
    # Récupérer les métriques finales
    metrics = trainer.callback_metrics
    results = {
        'model_name': model_name,
        'val_acc_ambiance': metrics.get('val_acc_Ambiance', 0).item(),
        'val_acc_cuisine': metrics.get('val_acc_Cuisine', 0).item(),
        'val_acc_prix': metrics.get('val_acc_Prix', 0).item(),
        'val_acc_service': metrics.get('val_acc_Service', 0).item(),
        'val_loss': metrics.get('val_loss', 0).item()
    }
    results['mean_val_acc'] = sum([
        results['val_acc_ambiance'],
        results['val_acc_cuisine'],
        results['val_acc_prix'],
        results['val_acc_service']
    ]) / 4.0
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=15)
    args = parser.parse_args()
    
    # Configuration de base
    cfg = ModelConfig(
        model_name="",  # sera remplacé
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backbone_lr=args.backbone_lr,
        max_epochs=args.max_epochs,
        device=args.device
    )
    
    # Charger les données
    train_data, val_data = load_data()
    
    # Test uniquement CamemBERT-base (large est déjà le baseline)
    models_to_test = [
        'camembert/camembert-base'
    ]
    
    all_results = []
    for model_name in models_to_test:
        results = train_model(model_name, cfg, train_data, val_data)
        all_results.append(results)
    
    # Afficher la comparaison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80 + "\n")
    
    for results in all_results:
        print(f"{results['model_name']}:")
        print(f"  Mean Val Acc: {results['mean_val_acc']:.4f}")
        print(f"  Ambiance: {results['val_acc_ambiance']:.4f}")
        print(f"  Cuisine:  {results['val_acc_cuisine']:.4f}")
        print(f"  Prix:     {results['val_acc_prix']:.4f}")
        print(f"  Service:  {results['val_acc_service']:.4f}")
        print(f"  Val Loss: {results['val_loss']:.4f}")
        print()
    
    # Sauvegarder
    output_dir = Path("model_comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "camembert_variants.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {output_dir}/camembert_variants.json")


if __name__ == "__main__":
    main()
