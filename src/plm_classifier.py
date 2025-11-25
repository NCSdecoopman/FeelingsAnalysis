import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from lightning import LightningModule
from config import Config


class AspectDataset(Dataset):
    """Dataset pour la classification multi-aspect"""
    
    def __init__(self, data: list[dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aspects = ['Prix', 'Cuisine', 'Service', 'Ambiance']
        self.label_map = {'Positive': 0, 'Négative': 1, 'Neutre': 2, 'NE': 3}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['Avis']
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels pour chaque aspect
        labels = torch.tensor([self.label_map[item[aspect]] for aspect in self.aspects])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


# The following function `run_project` is placed here as a standalone function
# because its content indicates it's not a method of `AspectDataset` despite
# its suggested placement in the instruction snippet.



class PLMClassifier(LightningModule):
    """Classificateur multi-aspect basé sur CamemBERT"""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.aspects = ['Prix', 'Cuisine', 'Service', 'Ambiance']
        self.label_map = {'Positive': 0, 'Négative': 1, 'Neutre': 2, 'NE': 3}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        self.tokenizer = AutoTokenizer.from_pretrained('camembert/camembert-large')
        # Disable the built‑in pooler (we use the CLS token directly)
        self.backbone = AutoModel.from_pretrained(
            'camembert/camembert-large',
            add_pooling_layer=False
        )
        # Enable gradient checkpointing to reduce memory usage
        self.backbone.gradient_checkpointing_enable()
        
        # CRITIQUE: Dégeler le backbone pour le fine-tuning complet
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # 4 têtes de classification (une par aspect)
        hidden_size = self.backbone.config.hidden_size
        self.classifiers = nn.ModuleDict({
            aspect: nn.Linear(hidden_size, 4)  # 4 classes: Positive, Négative, Neutre, NE
            for aspect in self.aspects
        })
        
        # Loss function avec label smoothing pour réduire l'overfitting
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.save_hyperparameters()
    
    def forward(self, input_ids, attention_mask):
        # Passage dans le backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Utiliser le [CLS] token (premier token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Prédictions pour chaque aspect
        logits = {
            aspect: self.classifiers[aspect](cls_output)
            for aspect in self.aspects
        }
        
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']  # Shape: [batch_size, 4]
        
        logits = self(input_ids, attention_mask)
        
        # Calculer la perte pour chaque aspect
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
        
        # Calculer la perte et l'accuracy pour chaque aspect
        total_loss = 0
        for i, aspect in enumerate(self.aspects):
            loss = self.criterion(logits[aspect], labels[:, i])
            total_loss += loss
            
            # Accuracy
            preds = torch.argmax(logits[aspect], dim=1)
            acc = (preds == labels[:, i]).float().mean()
            self.log(f'val_acc_{aspect}', acc, prog_bar=False)
        
        self.log('val_loss', total_loss, prog_bar=True)
        return total_loss
    
    def configure_optimizers(self):
        # Discriminative Learning Rates: backbone vs têtes
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
        
        # Scheduler with warmup and optional cosine/linear decay
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            if current_step < self.cfg.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.cfg.warmup_steps))
            else:
                # Post‑warmup schedule
                total_steps = max(1, self.trainer.estimated_stepping_batches)
                progress = (current_step - self.cfg.warmup_steps) / float(total_steps - self.cfg.warmup_steps)
                if self.cfg.scheduler == "cosine":
                    # Cosine annealing
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                else:
                    # Linear decay to zero
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
    
    def predict(self, text: str) -> dict[str, str]:
        """Prédire les opinions pour un texte donné"""
        self.eval()
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.cfg.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Déplacer vers le bon device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Prédiction
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
        
        # Convertir les logits en labels
        predictions = {}
        for aspect in self.aspects:
            pred_idx = torch.argmax(logits[aspect], dim=1).item()
            predictions[aspect] = self.reverse_label_map[pred_idx]
        
        return predictions
