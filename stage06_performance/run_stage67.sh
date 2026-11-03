#!/bin/bash
# 6단계 자원 측정 + 7단계 학습 전후 평가를 한 번에 실행 (웹캠 불필요)
# 결과: runs/stage6/<mode>.log, <mode>.tegra, runs/stage7/*.log
cd ~/eg2-lab && source ~/eg2/bin/activate
mkdir -p runs/stage6 runs/stage7

for mode in text image video audio all; do
    tegrastats --interval 500 --logfile runs/stage6/$mode.tegra &
    python measure.py gpu_load.py $mode > runs/stage6/$mode.log 2>&1
    tegrastats --stop 2>/dev/null; kill %1 2>/dev/null; wait 2>/dev/null
    sleep 3
done

python eval_ir.py                      > runs/stage7/base.log 2>&1
python eval_ir.py models/eg2-classroom > runs/stage7/finetuned.log 2>&1
echo DONE
