#!/bin/bash
# V2A Inspect 상시 구동 스크립트
#
# 사용법:
#   tmux new -s v2a_inspect
#   bash run.sh
#   # Ctrl+B, D 로 tmux detach (백그라운드 유지)
#
# 재접속:
#   tmux attach -t v2a_inspect
#
# 종료:
#   tmux kill-session -t v2a_inspect

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화 (venv 우선, 없으면 conda 시도)
if [ -f "$HOME/envs/v2a_ui-venv/bin/activate" ]; then
    source "$HOME/envs/v2a_ui-venv/bin/activate"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate v2a_ui
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate v2a_ui
fi

mkdir -p logs

RESTART_COUNT=0
MAX_RESTART_DELAY=60

echo "========================================"
echo "V2A Inspect 상시 구동 시작"
echo "시작 시간: $(date)"
echo "포트: 8503"
echo "========================================"

while true; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo ""
    echo "[$(date)] V2A Inspect 시작 (restart #$RESTART_COUNT)..."

    PYTHONUNBUFFERED=1 python run.py --port 8503 2>&1 | tee -a logs/inspect_forever.log
    EXIT_CODE=${PIPESTATUS[0]}

    echo "[$(date)] V2A Inspect 종료 (exit code: $EXIT_CODE)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "정상 종료. 재시작하지 않습니다."
        break
    fi

    DELAY=$((RESTART_COUNT < 6 ? RESTART_COUNT * 10 : MAX_RESTART_DELAY))
    echo "[$(date)] ${DELAY}초 후 재시작..."
    sleep $DELAY
done

echo "[$(date)] V2A Inspect 상시 구동 종료"
