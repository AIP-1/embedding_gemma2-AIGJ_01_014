#!/bin/bash
# 6단계 자원 측정 + 7단계 학습 전후 평가를 한 번에 실행 (웹캠 불필요)
# 결과: stage06_performance/results/<mode>.log, <mode>.tegra, stage07_finetune/results/*.log
cd "$(dirname "$0")" && source ~/eg2/bin/activate
mkdir -p results

for mode in text image video audio all; do
    tegrastats --interval 500 --logfile results/$mode.tegra &
    python measure.py gpu_load.py $mode > results/$mode.log 2>&1
    tegrastats --stop 2>/dev/null; kill %1 2>/dev/null; wait 2>/dev/null
    sleep 3
done

cd ../stage07_finetune && mkdir -p results
python eval_ir.py                      > results/base.log 2>&1
python eval_ir.py models/eg2-classroom > results/finetuned.log 2>&1
echo DONE
