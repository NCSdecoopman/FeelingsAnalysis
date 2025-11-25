from tqdm import tqdm
from torch.utils.data import DataLoader
from lightning import Trainer

from config import Config
from llm_classifier import LLMClassifier
from plm_classifier import PLMClassifier, AspectDataset

class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    METHOD: str = 'PLMFT'  # or 'LLM' (for Pretrained Language Model Fine-Tuning)

    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.METHOD == 'LLM':
            self.classifier = LLMClassifier(cfg)
        elif self.METHOD == 'PLMFT':
            self.classifier = PLMClassifier(cfg)


    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        :param train_data:
        :param val_data:
        :param device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut dire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu
        :return:
        """
        if self.METHOD == 'LLM':
            # Pas d'entraînement pour LLM en zero-shot
            return
        
        # Créer les datasets
        train_dataset = AspectDataset(train_data, self.classifier.tokenizer, self.cfg.max_length)
        val_dataset = AspectDataset(val_data, self.classifier.tokenizer, self.cfg.max_length)
        
        # Créer les dataloaders avec optimisations de performance
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=8,  # Increased for CPU parallelism
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.cfg.batch_size, 
            shuffle=False, 
            num_workers=8,  # Increased for CPU parallelism
            pin_memory=True,
            persistent_workers=True
        )
        
        # Early stopping callback
        from lightning.pytorch.callbacks import EarlyStopping
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.cfg.early_stopping_patience,
            mode='min',
            verbose=True
        )
        
        # Configurer le trainer avec Mixed Precision Training
        # Note: Multi-GPU DDP désactivé (problème NCCL driver version)
        trainer = Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator='gpu' if device >= 0 else 'cpu',
            devices=[device] if device >= 0 else 'auto',
            log_every_n_steps=10,
            enable_checkpointing=False,
            callbacks=[early_stop_callback],
            accumulate_grad_batches=self.cfg.accumulate_grad_batches,  # Use config value for gradient accumulation
            precision='16-mixed'  # Mixed Precision Training pour accélération ~2-3x
        )
        
        # Entraîner le modèle
        trainer.fit(self.classifier, train_loader, val_loader)



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        :param texts:
        :param device: device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu à utiliser
        :return:
        """
        all_opinions = []
        
        if self.METHOD == 'LLM':
            # Pour LLM, traiter un à un
            for text in tqdm(texts):
                opinions = self.classifier.predict(text)
                all_opinions.append(opinions)
        else:
            # Pour PLMFT, traiter un à un aussi (peut être optimisé par batch)
            for text in tqdm(texts):
                opinions = self.classifier.predict(text)
                all_opinions.append(opinions)
        
        return all_opinions








