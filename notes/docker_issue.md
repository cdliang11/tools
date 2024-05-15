
# docker相关的问题记录


## docker共享内存设置
`shm-size`参数用于设置 Docker 容器中 /dev/shm 的大小。默认情况下，大多数 Linux 系统的大小是64M，但在处理一些需要大量内存的任务时，可能会出现不足的情况。

在 Docker 中设置 `shm-size`` 的方法是在启动容器时使用 --shm-size 选项。例如：
```bash
docker run -d --shm-size="256m" myimage
```
这将为运行的容器设置共享内存大小为256MB。

如果你使用的是 Docker Compose，可以在 docker-compose.yml 文件中设置：
```bash
version: '3'
services:
  myservice:
    image: myimage
    shm_size: '256mb'
```
请注意，设置的大小可能需要是 KB、MB、GB 的格式，并且大小必须是整数，不能有小数.


## docker跑tensorrt-llm 多卡并行报Signal code: Non-existant physical address (2)错误
```bash
[2213e8cf917e:00309] *** Process received signal ***
[2213e8cf917e:00309] Signal: Bus error (7)
[2213e8cf917e:00309] Signal code: Non-existant physical address (2)
[2213e8cf917e:00309] Failing at address: 0x7f30b0f92000
[2213e8cf917e:00310] *** Process received signal ***
[2213e8cf917e:00310] Signal: Bus error (7)
[2213e8cf917e:00310] Signal code: Non-existant physical address (2)
[2213e8cf917e:00310] Failing at address: 0x7f56216e1000
[2213e8cf917e:00311] *** Process received signal ***
[2213e8cf917e:00311] Signal: Bus error (7)
[2213e8cf917e:00311] Signal code: Non-existant physical address (2)
[2213e8cf917e:00311] Failing at address: 0x7f458a191000
[2213e8cf917e:00312] *** Process received signal ***
[2213e8cf917e:00312] Signal: Bus error (7)
[2213e8cf917e:00312] Signal code: Non-existant physical address (2)
[2213e8cf917e:00312] Failing at address: 0x7fd8010a5000
```
大概率是docker共享内存不够。ref : https://github.com/NVIDIA/TensorRT-LLM/issues/963

查看docker共享内存的指令:
```bash
df -h | grep shm
```

