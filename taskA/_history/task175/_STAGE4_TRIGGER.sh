#!/bin/bash
PID=$(cat /home/wlia0047/ar57/wenyu/GeneRec/products/task175/_TRAINING_PID 2>/dev/null)
if [ -z "$PID" ]; then
    echo "[$(date)] ❌ _TRAINING_PID not found for #175" >> /home/wlia0047/ar57/wenyu/GeneRec/logs/task175/_stage4_trigger.log
    exit 1
fi
LOG=/home/wlia0047/ar57/wenyu/GeneRec/logs/task175/_stage4_trigger.log
echo "[$(date)] Task #175 Stage 4 daemon watching PID=$PID" >> $LOG
while ps -p $PID > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] Training PID $PID 退出, 启动 Stage 4..." >> $LOG
bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task175_t5mini_orc_locked_stage4_eval.sh
STAGE4_EXIT=$?
echo "[$(date)] Stage 4 exit: $STAGE4_EXIT" >> $LOG
