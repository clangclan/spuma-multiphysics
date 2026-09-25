set -e
cd /home/jsw/문서/analysis/spuma-multiphysics-gpu-opt
for t in 0.0002:/home/jsw/문서/analysis/runs/n160_40bar_200to500us_gpu_20260925/production/40bar/to-300us/0.0002 0.0005:/home/jsw/문서/analysis/runs/n160_40bar_200to500us_gpu_20260925/production/40bar/to-500us/0.0005; do
 tag=${t%%:*}; ck=${t#*:}
 for sel in noncondensable condensable liquid curved; do
  for off in 0 1048576; do
   [ $sel != condensable ] && [ $off != 0 ] && continue
   /home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python tools/hem_checkpoint_batch.py $ck --model-from /home/jsw/문서/analysis/runs/precision_revisit_20260924/hem-40bar.bin --sigma 0.0020577273399028486 --select $sel --limit 1048576 --offset $off --analytic-jacobian 1 --stable-gas-prune 1 --output /tmp/claude-1000/-home-jsw----analysis/c01a763b-bfdf-49a0-9678-3b18bb851fea/scratchpad/r500/$tag-$sel-$off.bin 2>&1 | tail -1 | sed "s/^/$tag $sel $off: /" || true
   [ -f /tmp/claude-1000/-home-jsw----analysis/c01a763b-bfdf-49a0-9678-3b18bb851fea/scratchpad/r500/$tag-$sel-$off.bin ] && /home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python tools/replay_hem_precision.py /tmp/claude-1000/-home-jsw----analysis/c01a763b-bfdf-49a0-9678-3b18bb851fea/scratchpad/r500/$tag-$sel-$off.bin --library /home/jsw/문서/analysis/runs/n160_40bar_200to500us_gpu_20260925/runtime/lib/libreactiveTransport.so --output /tmp/claude-1000/-home-jsw----analysis/c01a763b-bfdf-49a0-9678-3b18bb851fea/scratchpad/r500/rep-$tag-$sel-$off --repeats 3 2>&1 | tail -2
  done
 done
done
