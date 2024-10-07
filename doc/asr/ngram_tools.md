# 【语音识别】ngram常用工具（实战篇）

上篇文章介绍了ngram的基础原理，并手动计算了一个简单的ngram模型。本文将介绍两个常用的ngram训练工具：`kenlm / srilm`，方便实际生产中使用。

- `kenlm`主要用来训练得到ngram模型，速度相对较快，并且使用简单。
- `srilm`是一个功能更全的工具，主要使用其中`ngram，ngram-count`两个命令，可以完成模型训练、模型剪枝、模型合并。

<u>实际中两个工具混合起来使用。</u>

## Kenlm
### 安装
GitHub链接：https://github.com/kpu/kenlm/blob/master/BUILDING

将源码clone到本地：
```bash
git clone https://github.com/kpu/kenlm
```

### 使用cmake编译

```bash
# 安装第三方依赖
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev

# 源码编译
mkdir -p build
cd build
cmake ..
make -j 4
```

### 训练语言模型
kenlm常用的命令是lmplz，很方便完成ngram模型训练，得到arpa文件

准备语料：

```
中国
爱 中国
我
爱
中国
```

### 训练模型

```bash
 ./build/bin/lmplz -o 2 --verbose_header --text ./corpus.txt --arpa ./demo.arpa
```

> bin/lmplz 命令的参数解释：
>
>   -o: 语言模型的阶数 （上文中的 N）
>
>   --verbose_header: 在文件头加上统计信息
>
>   --text: 存放训练语料的文件
>
>   --arpa: 输出的 arpa 文件名
>
>   -S[--memory]arg(=80%)Sortingmemory： 内存预占用量
>



**训练log输出：**
```
=== 1/5 Counting and sorting n-grams ===
Reading /home/liangcd/github/kenlm/corpus.txt
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
****************************************************************************************************
Unigram tokens 6 types 6
=== 2/5 Calculating and sorting adjusted counts ===
Chain sizes: 1:72 2:6397437542
Statistics:
1 6 D1=0.5 D2=0.5 D3+=3
2 7 D1=0.5 D2=1.25 D3+=3
Memory estimate for binary LM:
type       B
probing  292 assuming -p 1.5
probing  320 assuming -r models -p 1.5
trie     226 without quantization
trie    1235 assuming -q 8 -b 8 quantization
trie     226 assuming -a 22 array pointer compression
trie    1235 assuming -a 22 -q 8 -b 8 array pointer compression and quantization
=== 3/5 Calculating and sorting initial probabilities ===
Chain sizes: 1:72 2:112
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
####################################################################################################
=== 4/5 Calculating and writing order-interpolated probabilities ===
Chain sizes: 1:72 2:112
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
####################################################################################################
=== 5/5 Writing ARPA model ===
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
****************************************************************************************************
Name:lmplz      VmPeak:6411524 kB       VmRSS:6044 kB   RSSMax:1702800 kB       user:0.080416   sys:0.593073    CPU:0.673714    real:0.675392
```

训练完成后会得到arpa文件，内容如下：

```
# Input file: /home/liangcd/github/kenlm/corpus.txt
# Token count: 6
# Smoothing: Modified Kneser-Ney
\data\
ngram 1=6
ngram 2=7

\1-grams:
-0.89085555     <unk>   0
0       <s>     -0.22184873
-0.89085555     </s>    0
-0.46488678     中国    0
-0.69896996     爱      -0.30103
-0.69896996     我      -0.30103

\2-grams:
-0.89085555     中国 </s>
-0.50267535     爱 </s>
-0.24850096     我 </s>
-0.44889864     <s> 中国
-0.37527603     爱 中国
-0.56863624     <s> 爱
-0.6575773      <s> 我

\end\
```

**和上篇文章中手动计算的结果一致。**

## srilm

srilm是一个ngram的c++工具库，主要用来实现ngram相关的算法。语音识别中ngram语言模型训练过程中基本用的工具有两个：ngram-count ，ngram。支持模型训练和模型评估

### 安装

1. github上有最新的安装包：`https://github.com/gsayer/SRILM`，将压缩包下载到本地，并解压

2. 修改Makefile文件，将`# SRILM = /home/speech/stolcke/project/srilm/devel` 替换为`SRILM = /path/to/srilm`

3. 编译：

    ```bash
    sudo tcsh
    sudo make NO_TCL=1 MACHINE_TYPE=i686-m64 World
    sudo ./bin/i686-m64/ngram-count -help  # 检查是否编译成功
    ```

4. 设置环境变量

    ```bash
    SRILM=/path/to/srilm
    MACHINE_TYPE=i686-m64
    export PATH=.:$PATH:$SRILM/bin/$MACHINE_TYPE:$SRILM/bin
    ```

### 模型训练

使用ngram-count命令训练模型，以下是参数解析

```bash
ngram-count
##功能
#读取分词后的text文件或者count文件，然后用来输出最后汇总的count文件或者语言模型
##参数
#输入文本：
#  -read 读取count文件
#  -text 读取分词后的文本文件
#词典文件：
#  -vocab 限制text和count文件的单词，没有出现在词典的单词替换为<unk>；
#         如果没有加该选项，所有的单词将会被自动加入词典
#  -limit-vocab 只限制count文件的单词（对text文件无效）；
#               没有出现在词典里面的count将会被丢弃
#  -write-vocab 输出词典
#语言模型：
#  -lm 输出语言模型
#  -write-binary-lm 输出二进制的语言模型
#  -sort 输出语言模型gram排序
```

两种训练方法：

```bash
#方法1: text->count->lm
ngram-count -text $text -vocab ${vocab} -order 2 -write {count}
ngram-count -read ${count} -vocab ${vocab} -order 2 -lm ${arpa} -interpolate -kndiscount ##改进的Kneser-ney平滑算法

#方法2: text->lm
ngram-count -text ${text} -vocab ${vocab} -order 2 -lm ${arpa} -interpolate -kndiscount
```

### ppl测试

```bash
ngram
##功能
#用于评估语言模型的好坏，或者是计算特定句子的得分，用于语音识别的识别结果分析。
##参数
#计算得分：
#  -order 模型阶数，默认使用3阶
#  -lm 使用的语言模型
#  -ppl 后跟需要打分的句子（一行一句，已经分词），ppl表示所有单词，ppl1表示除了</s>以外的单词
#    -debug 0 只输出整体情况
#    -debug 1 具体到句子
#    -debug 2 具体每个词的概率
#产生句子：
#  -gen 产生句子的个数
#  -seed 产生句子用到的random seed
ngram -lm ${lm} -order 2 -ppl ${file} -debug 1 > ${ppl}
```

### 模型剪枝

```bash
ngram
##功能
#用于减小语言模型的大小，剪枝原理参考(http://blog.csdn.net/xmdxcsj/article/details/50321613)
##参数
#模型裁剪：
#  -prune threshold 删除一些ngram，满足删除以后模型的ppl增加值小于threshold，越大剪枝剪得越狠
#  -write-lm 新的语言模型
ngram -lm ${oldlm} -order 2 -prune ${thres} -write-lm ${newlm}
```

### 模型合并
```bash
ngram
##功能
#用于多个语言模型之间插值合并，以期望改善模型的效果
##参数
#模型插值：
#  -mix-lm 用于插值的第二个ngram模型，-lm是第一个ngram模型
#  -lambda 主模型（-lm对应模型）的插值比例，0~1，默认是0.5
#  -mix-lm2 用于插值的第三个模型
#  -mix-lambda2 用于插值的第二个模型（-mix-lm对应的模型）的比例，
#               那么第二个模型的比例为1-lambda-mix-lambda2
#  -vocab 当两个模型的词典不一样的时候，使用该参数限制词典列表，没有效果
#  -limit-vocab 当两个模型的词典不一样的时候，使用该参数限制词典列表，没有效果
ngram -lm ${mainlm} -order 2 -mix-lm ${mixlm} -lambda 0.8 -write-lm ${mergelm}
```

## 参考
- [srilm安装](https://blog.csdn.net/weixin_44253298/article/details/123737481?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-8-123737481-blog-136168163.235^v43^pc_blog_bottom_relevance_base9&spm=1001.2101.3001.4242.5&utm_relevant_index=11)
- https://zhuanlan.zhihu.com/p/273606445
