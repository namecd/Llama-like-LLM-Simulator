CUDA_VISIBLE_DEVICES=3 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  -o simulate-256.nsys-rep \
  python easy_simulate.py