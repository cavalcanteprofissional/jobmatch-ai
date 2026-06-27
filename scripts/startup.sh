#!/usr/bin/env bash
# startup.sh — JobMatch AI
# Executado na inicialização do container.
# Se `data/` estiver vazio (primeiro deploy), baixa datasets e treina modelos.
# Se `data/models/` existe, apenas inicia o servidor.
set -euo pipefail

echo "=== JobMatch AI Startup ==="

MODELS_DIR="/app/data/models"
PROCESSED_DIR="/app/data/processed"

# ── Verificar se modelos já existem ──────────────────────────
if [ -f "$MODELS_DIR/metrics.json" ] && [ -f "$MODELS_DIR/classifier.pkl" ]; then
    echo "✓ Modelos encontrados. Iniciando servidor..."
    exec uvicorn src.api.server:app --host 0.0.0.0 --port 8000
fi

echo "! Nenhum modelo encontrado. Iniciando pipeline de treino..."

# ── Verificar se dados brutos existem ────────────────────────
if [ ! -d "$PROCESSED_DIR" ] || [ -z "$(ls -A "$PROCESSED_DIR" 2>/dev/null)" ]; then
    echo "→ Dados processados não encontrados. Executando compose_datasets..."
    python -c "from src.pipeline.compose_datasets import main; main()"
    echo "✓ Dados processados criados."
else
    echo "✓ Dados processados já existem."
fi

# ── Pipeline de treino rápido ────────────────────────────────
echo "→ Executando train_pipeline.py..."
python train_pipeline.py
echo "✓ Modelo base treinado."

# ── Relatório de avaliação (nested CV ~10 min) ───────────────
echo "→ Executando reload_eval.py (nested CV)..."
python scripts/reload_eval.py
echo "✓ Avaliação concluída."

# ── Iniciar servidor ─────────────────────────────────────────
echo "✓ Pipeline concluída. Iniciando servidor..."
exec uvicorn src.api.server:app --host 0.0.0.0 --port 8000
