#!/bin/bash
# Gunluk kullanim profili - RTX 5080 16GB / Qwen3.8-27B UD-Q3_K_XL
# Olculmus: ~106 t/s generation, 15605 MiB peak (tavan 15841 CUDA-visible)
_M="$MODEL"
source ~/llm-bench/tools/model.env
[ -n "$_M" ] && MODEL="$_M"   # disaridan MODEL= verilirse o kazanir
exec ~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  -a "${ALIAS:-Qwen3.8-27B-UD-Q3_K_XL}" \
  -ngl 99 \
  -fit off \
  -c ${CTX:-90112} \
  -ctk q4_0 -ctv q4_0 \
  -fa on \
  --spec-type draft-mtp --spec-draft-n-max ${NMAX:-3} \
  -np 1 \
  --jinja \
  --metrics \
  --host ${HOST:-0.0.0.0} --port ${PORT:-8080} \
  "$@"
