# eagle

[benchmark_latency](./benchmark_latency.py)用来测试eagle的速度

使用方法：
```bash
# 1. 下载官方代码
git clone https://github.com/SafeAILab/EAGLE.git
cp benchmark_latency.py EAGLE
cd EAGLE

# 2. 测试

python benchmark_latency_a800.py \
  --ea-model-path ${EAGLE-mixtral-instruct-8x7B} \
  --base-model-path ${Mixtral-8x7B-Instruct-v0.1} \
  --model-type mixtral \
  --out_res ./out.res \
  --useEaInfer
```
