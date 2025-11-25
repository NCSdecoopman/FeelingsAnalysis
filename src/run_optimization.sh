#!/bin/bash
# Script pour lancer l'optimisation Optuna en tâche de fond

# Configuration
N_TRIALS=${1:-50}
DEVICE=${2:-0}
N_EPOCHS=${3:-10}
LOGFILE="optimization_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOGFILE"
echo "Hyperparameter Optimization avec Optuna" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "N_TRIALS: $N_TRIALS" | tee -a "$LOGFILE"
echo "DEVICE: GPU $DEVICE" | tee -a "$LOGFILE"
echo "N_EPOCHS per trial: $N_EPOCHS" | tee -a "$LOGFILE"
echo "LOGFILE: $LOGFILE" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# Lancer l'optimisation
python3 optimize_hyperparams.py \
    --n-trials $N_TRIALS \
    --device $DEVICE \
    --n-epochs $N_EPOCHS \
    2>&1 | tee -a "$LOGFILE"

# Notifier la fin
echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "Optimization terminée !" | tee -a "$LOGFILE"
echo "Résultats dans: optimization_results/" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
