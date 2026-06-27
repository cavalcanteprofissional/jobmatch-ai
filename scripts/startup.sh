#!/usr/bin/env bash
# startup.sh — JobMatch AI
# Executado na inicialização do container.
# Modelos são embutidos na imagem Docker (free tier não tem RAM para treinar).
# Se `data/models/` existe, inicia o servidor. Caso contrário, loga erro.
set -euo pipefail

echo "=== JobMatch AI Startup ==="

MODELS_DIR="/app/data/models"

if [ -f "$MODELS_DIR/metrics.json" ] && [ -f "$MODELS_DIR/classifier.pkl" ]; then
    echo "✓ Modelos encontrados."
    echo "→ Testando import do app..."
    python -c "from src.api.server import app; print('✓ App importado com sucesso')"
    echo "→ Iniciando servidor..."
    exec uvicorn src.api.server:app --host 0.0.0.0 --port 8000
fi

echo "✗ Nenhum modelo encontrado em $MODELS_DIR."
echo "  O build da imagem Docker deve incluir data/models/ e data/processed/."
echo "  Veja README.md → 'Deploy Free Tier (Render.com)' para instruções."
echo "  Iniciando servidor sem modelos (health check responderá models_loaded=false)."
exec uvicorn src.api.server:app --host 0.0.0.0 --port 8000
