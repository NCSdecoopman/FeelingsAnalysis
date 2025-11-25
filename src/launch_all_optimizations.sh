#!/bin/bash
# Script maître pour lancer toutes les optimisations en parallèle

echo "========================================================================"
echo "  FeelingsAnalysis - Optimization Master Script"
echo "  Target: 93%+ accuracy"
echo "========================================================================"
echo ""

# Créer dossier pour les logs
mkdir -p optimization_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Fonction pour logger
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "optimization_logs/master_${TIMESTAMP}.log"
}

log "Starting optimization pipeline..."
log "Available GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"

# ======================================================================
# Phase 1: Hyperparameter Optimization avec Optuna (50 trials)
# ======================================================================
log "Phase 1: Launching Optuna hyperparameter optimization (50 trials)"
nohup bash run_optimization.sh 50 0 10 > optimization_logs/optuna_${TIMESTAMP}.log 2>&1 &
OPTUNA_PID=$!
echo $OPTUNA_PID > optimization_logs/optuna.pid
log "Optuna launched with PID: $OPTUNA_PID"

# Attendre un peu avant de lancer la suite
sleep 5

# ======================================================================
# Phase 2: Test CamemBERT-base vs Large en parallèle
# ======================================================================
log "Phase 2: Launching CamemBERT variant comparison"
nohup python3 compare_camembert_variants.py \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --backbone-lr 1e-5 \
    --max-epochs 15 \
    > optimization_logs/camembert_comparison_${TIMESTAMP}.log 2>&1 &
CAMEMBERT_PID=$!
echo $CAMEMBERT_PID > optimization_logs/camembert_comparison.pid
log "CamemBERT comparison launched with PID: $CAMEMBERT_PID"

# ======================================================================
# Monitoring
# ======================================================================
log ""
log "========================================================================"
log "All tasks launched successfully!"
log "========================================================================"
log ""
log "Running processes:"
log "  - Optuna optimization: PID $OPTUNA_PID"
log "  - CamemBERT comparison: PID $CAMEMBERT_PID"
log ""
log "To monitor progress:"
log "  - Optuna: tail -f optimization_logs/optuna_${TIMESTAMP}.log"
log "  - CamemBERT: tail -f optimization_logs/camembert_comparison_${TIMESTAMP}.log"
log ""
log "To stop all:"
log "  - kill $OPTUNA_PID $CAMEMBERT_PID"
log ""
log "Results will be saved in:"
log "  - optimization_results/"
log "  - model_comparison_results/"
log ""

# Créer un fichier de suivi
cat > optimization_logs/running_jobs_${TIMESTAMP}.txt <<EOF
Optimization Jobs Started: $(date)

Optuna Hyperparameter Search:
  PID: $OPTUNA_PID
  Log: optimization_logs/optuna_${TIMESTAMP}.log
  Status: tail -f optimization_logs/optuna_${TIMESTAMP}.log

CamemBERT Variants Comparison:
  PID: $CAMEMBERT_PID
  Log: optimization_logs/camembert_comparison_${TIMESTAMP}.log
  Status: tail -f optimization_logs/camembert_comparison_${TIMESTAMP}.log

To check if jobs are still running:
  ps -p $OPTUNA_PID,$CAMEMBERT_PID

To stop all jobs:
  kill $OPTUNA_PID $CAMEMBERT_PID
EOF

log "Job tracker saved to: optimization_logs/running_jobs_${TIMESTAMP}.txt"
log "========================================================================"

# Attendre la fin des jobs (optionnel, commenté par défaut)
# wait $OPTUNA_PID $CAMEMBERT_PID
# log "All optimization jobs completed!"
