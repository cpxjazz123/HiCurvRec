#!/bin/bash
# Task #80/#81 数据重建: 从 McAuley 5-core 重新下载 Musical_Instruments
# 数据已被删, 必须重建用于 FDSA / S³Rec / P5-CID / P5-SID 训练

set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/data/amazon_data/

mkdir -p musical_instruments
cd musical_instruments

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task80_81_data_rebuild_${TS}.log"

echo "===== [Task #80/81] Musical_Instruments data rebuild started at $(date) =====" | tee "$LOG_FILE"

# 使用 gdown 或 wget 从 McAuley 仓库下载 5-core 数据
# 5-core Musical_Instruments: https://nijianmo.github.io/amazon/index.html
# 5-core (2018): https://datacloud.di.to.it.ac.at/?prefix=amazon/5core/
# 直接 URL: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz

if [ ! -f "Musical_Instruments_5.json.gz" ]; then
    echo "Downloading Musical_Instruments 5-core from McAuley ..." | tee -a "$LOG_FILE"
    wget -q --show-progress https://datacloud.di.to.it.ac.at/amazon/5core/Musical_Instruments_5.json.gz 2>&1 | tee -a "$LOG_FILE" || \
    wget -q --show-progress https://nijianmo.github.io/amazon-index/v2/data/Musical_Instruments_5.json.gz 2>&1 | tee -a "$LOG_FILE"
fi

ls -la Musical_Instruments_5.json.gz | tee -a "$LOG_FILE"

echo "===== [Task #80/81] Data download completed at $(date) =====" | tee -a "$LOG_FILE"
