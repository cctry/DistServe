HOST="0.0.0.0"
PORT=8000

MODEL=facebook/opt-66b


CONTEXT_TP=$3
CONTEXT_PP=$4
DECODING_TP=$5
DECODING_PP=$6


MAX_NUM_BLOCKS=$((65536 / 16)) # Maximum total length 65536


python -m distserve.api_server.distserve_api_server \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --context-tensor-parallel-size "$CONTEXT_TP" \
    --context-pipeline-parallel-size "$CONTEXT_PP" \
    --decoding-tensor-parallel-size "$DECODING_TP" \
    --decoding-pipeline-parallel-size "$DECODING_PP" \
    --block-size 16 \
    --max-num-blocks-per-req "$MAX_NUM_BLOCKS" \
    --gpu-memory-utilization 0.95 \
    --swap-space 16 \
    --context-sched-policy fcfs \
    --context-max-batch-size 1024 \
    --context-max-tokens-per-batch 65536 \
    --decoding-sched-policy fcfs \
    --decoding-max-batch-size 1024 \
    --decoding-max-tokens-per-batch 65536 \
    # --use-dummy-weights # Not tested