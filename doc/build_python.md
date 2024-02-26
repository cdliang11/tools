# Python 及三方库 交叉编译 (ARM)

python交叉编译分为两部分：
1. 主机端的python-host  （电脑需要和设备端安装相同版本的python）
2. 目标设备端的python-target
推荐在hadoop046机器上进行交叉编译，环境配置全面

## 主机端python编译 (可选)
> 如果本地已经有对应版本的python，则跳过这一步
1. 从官网下载对应版本的源码并下载解压
https://www.python.org/downloads/source/
```
# 进入下载并解压的python路径下
cd your_python_download_path
./configure --prefix=$PWD/build_pc

make -j4 & make install
```

## 设备端python编译
### 第三方库交叉编译
1. Zlib
  下载一个你喜欢的 zlib版本：http://www.zlib.net/fossils/
```
wget http://zlib.net/zlib-1.3.tar.gz
tar -zxvf zlib-1.3.tar.gz
cd zlib-1.3/

export CC=aarch64-linux-gnu-gcc
export CROSS_PREFIX=/home/voice_gcc/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-
export CFLAGS="-Wall -g"

./configure --prefix=$PWD/zlib_arm

make -j4 && make install
```

2. libffi
  1. 下载一个你喜欢的 libffi版本：https://github.com/libffi/libffi/releases

```bash
wget https://github.com/libffi/libffi/releases/download/v3.4.4/libffi-3.4.4.tar.gz
tar -zxvf libffi-3.4.4.tar.gz
cd libffi-3.4.4

# export CROSS_PREFIX=/home/voice_gcc/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-

./configure \
--prefix=$PWD/ffi_arm \
--build=aarch64 \
--host=aarch64-linux-gnu \
--target=aarch64-linux-gnu \
CC=aarch64-linux-gnu-gcc \
CXX=aarch64-linux-gnu-g++ \
RANLIB=aarch64-linux-gnu-ranlib \
STRIP=aarch64-linux-gnu-strip \
AR=aarch64-linux-gnu-ar \
CFLAGS="-Wall -g" \
CPPFLAGS="-Wall -g"

make -j4 & make install
```

### 板端 python3.8 安装
如果是从源码编译了主机python，需要将源码编译的 Python 加入环境变量
`export PATH={your_python_path}/build_pc/bin/:$PATH`
1. 采用交叉编译的方式
```bash
cd ${your_python_path}

./configure \
--prefix=$PWD/build_arm \
--host=aarch64-linux-gnu \
--disable-ipv6 \
CC=aarch64-linux-gnu-gcc \
CXX=aarch64-linux-gnu-g++ \
RANLIB=aarch64-linux-gnu-ranlib \
STRIP=aarch64-linux-gnu-strip \
AR=aarch64-linux-gnu-ar \
LDFLAGS="-L{your_zlib_path}/zlib_arm/lib -L{your_libffi_path}/ffi_arm/lib" \
LIBS="-lz -lffi" \
CFLAGS="-Wall -g -I{your_zlib_path}/zlib_arm/include -I{your_libffi_path}/ffi_arm/include" \
CPPFLAGS="-Wall -g -I{your_zlib_path}/zlib_arm/include -I{your_libffi_path}/ffi_arm/include" \
ac_cv_file__dev_ptmx=no \
ac_cv_file__dev_ptc=no

make -j4 & make install
```

2. 板端测试
- 在下载到开发板上之前，需要把 zlib/libffi 对应的动态库先放到 Python lib/python3.x/lib-dynload 文件夹内，后续会依赖会用到。
```bash
cp {your_zlib_path}/zlib_arm/lib/libz.so* {your_python_path}/build_arm/lib/python3.x/lib-dynload/
cp {your_ffi_path}/zlib_arm/lib/libffi.so* {your_python_path}/build_arm/lib/python3.x/lib-dynload/
```
- 指定python路径
```bash
export PYTHON_ROOT=/userdata/tools/python3.8

export PATH=$PYTHON_ROOT/bin:$PATH

export LD_LIBRARY_PATH=$PYTHON_ROOT/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHON_ROOT:$PYTHON_ROOT/lib/:$PYTHON_ROOT/lib/python3.8:$PYTHON_ROOT/lib/python3.8/site-packages

export PYTHONHOME=$PYTHON_ROOT
```


## python库安装
> 核心思路：虚拟机上或者一个可以连网并且装了pip的arm设备上（比如x3 pi开发板），安装需要的库，然后传到目标板端的site-packages路径下

### 可pip的ARM设备安装第三方包
如果是为了在J5板端配置python环境，并安装tokenizer相关的package，直接从x3板端的环境中copy相关的包。
> x3 板子： 10.110.10.80
> J5 已配置好的环境：/userdata/tools/python3.8

### PC端虚拟环境交叉编译
#### 交叉编译环境配置
电脑端 Python 安装交叉编译库 cross_env， 这个库非常重要，后续所有交叉编译的 python 库都会用到。
1. 安装 cross_env
```bash
./{your_python_path}/build_pc/bin/python3 -m pip install cross_env
```
2. 开启 cross_env虚拟环境，执行完后当前目录下会生成一个名为 cross_venv的文件夹
```bash
{your_python_path}/build_pc/bin/python3 -m crossenv --without-pip {your_python_path}/build_arm/bin/python3 cross_venv
```
注：第一个为电脑端 Python 的路径（build_pc），第二个为设备端 Python 的路径（build_arm），勿搞错。
3. 激活虚拟环境
```bash
cd cross_env/cross/bin
source activate
```
4. 在虚拟环境中指定交叉编译工具链及编译参数的相关环境变量
```bash
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
export CFLAGS="-Wall -g"export CPPFLAGS="-Wall -g"
```
5. 检查环境是否正常
```bash
python -V #返回交叉编译的 python 版本号
which python #返回 cross_env 生成的虚拟环境下 python 的路径，如：cross_venv/cross/bin/python
```
6. 安装 pip
```bash
# 下载 get-pip
curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py -k
# 安装 pip
python get-pip.py
# 检查 pip 是否安装成功
python -m pip -V # 返回 pip 版本号
```
至此，交叉编译虚拟环境已配置完成，后续所有交叉编译步骤都是在虚拟环境激活的情况下操作。

#### 交叉编译示例
以numpy为例
1. pip 安装 numpy，这里指定了版本号，也可以安装你们自己喜欢的版本号
```bash
python -m pip install numpy==1.18.5
```
2. 将 numpy 包拷贝至设备端 python 的 site-packages/路径下
```bash
# 显示下电脑端刚交叉编译的 numpy 的安装路径，一般在 cross_venv/cross/lib/python3.7/site-packages 路径下
python -m pip show numpy
cp cross_venv/cross/lib/python3.7/site-packages/numpy* {your_python_path}/build_arm/lib/python3.x/site-packages/
```
3.  在开发板上验证下 numpy 是否安装成功正常运行
```bash
# 该步骤在开发板上运行
python3 -c "import numpy;print(numpy.__version__)" # 返回 numpy 版本号 1.18.5
```
