from dataclasses import dataclass

# par exemple: pour lancer tester votre rendu programme  avec 2 runs et en utilisant la gpu
# numéro 0, il suffit de taper la commande suivante:
#
#           python runproject.py --n_runs=2 --device=0

@dataclass
class Config:
    # General options
    device: int = 0
    ollama_url: str = 'http://localhost:11434'
    n_runs: int = 5
    # n_train is the number of samples on which to run the eval. n_trian=-1 means eval on all test data,
    n_train: int = -1
    # n_test is the number of samples on which to run the eval. n_test=-1 means eval on all test data,
    n_test: int = -1
    # PLM Fine-Tuning hyperparameters
    batch_size: int = 32  # updated per version_2
    accumulate_grad_batches: int = 4  # unchanged
    learning_rate: float = 2e-5  # updated lower LR
    backbone_lr: float = 1.0e-05    # unchanged
    max_epochs: int = 20  # increased epochs
    max_length: int = 256  # unchanged
    weight_decay: float = 0.01  # unchanged
    warmup_steps: int = 1000  # increased warmup steps
    scheduler: str = "linear"  # new field, default linear scheduler
    early_stopping_patience: int = 3  # new field for early stopping
