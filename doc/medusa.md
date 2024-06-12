
# [投机采样系列二] Medusa
> Paper: https://arxiv.org/abs/2401.10774.pdf
> Code: https://github.com/FasterDecoding/Medusa
> 加速比：2.2× ∼ 2.3×

Medusa不需要额外的小模型来生成候选序列。
- draft：使用medusa head，并行预测对应位置的topk个token
- Verify:  使用tree attention机制，一次LLM forward验证多个候选序列
- Accept：分为两种：greedy 和 typical acceptance

模型架构
Medusa 在现有模型基础上，增加多个Medusa Head，与原模型上的LM Head 一同做预测。
新增的Medusa Head 包含Block(可以多个堆叠)和分类头，输入为backbone模型的Last Hidden数据，输出为预测Token的概率
$$p_t^{(k)} = softmax(W_2^{(k)}(SiLU(W_1^{(k)} h_t) + h_t))$$
原文注释: We initialize W1(k) identically to the original language model head, and W2(k) to zero
下图左边有3个medusa头，包含原LM_head模型一次性可以输出1+3个token
[图片]
暂时无法在飞书文档外展示此内容

训练
Medusa-1: Frozen Backbone
Medusa1在训练过程中会将原模型参数冻结， Medusa 头参数需要训练
设 $$y_{t+k+1}$$ 为 t+k+1位置的token，训练loss为:
$$L_{Medusa-1} = \sum_{k=1} ^ {k} -\lambda_k \log p_t^{(k)}(y_{t+k+1})$$
常规的标签需要偏移1个位置, 这里的labels[..., 2 + i :] , 由于不训练 LM Head，所以偏移2个位置

Medusa-2: Joint Training
为了进一步提高Medusa head 的预测效率，Medusa-2采用联合训练的方式，有三种策略:
1. Combined loss: 在训练Medusa-1时，同时加入backbone模型也进行参数训练，即增加next-token预测的loss
2. Differential learning rates: 由于backbone model已经充分训练了，而medusa需要更充分的训练，设定不同的learning rate可以达到目的
3. Head Warmup （论文中指出 这种方法能有较好的效果）
one-stage：先训练base model作为medusa-1， 使用较少的epochs
   two-stage：参照1的方式，进行混合训练，这里提到的warmup指逐步增大$$\lambda$$

额外补充：self-distillation
仅靠SFT训练，Medusa头只能适应给定的数据集的分布，不能保证和base model的分布一致。
原因：在实际训练Medusa模型时，可能没有与目标模型输出分布相匹配的训练数据集。这可能是因为只发布了模型而没有训练数据，或者模型已经经过强化学习与人工反馈强化过程，使得模型的输出分布与训练数据集不同。

那么这里可以采用self-distillation的方式使得Medusa头适应base model的分布
具体来说使用模型本身生成Medusa头部的训练数据集，这个数据集与模型的输出分布相匹配

推理
在Speculative sampling 中, 使用 Draft model 来低成本获得Next-Next-Token , 而Medusa仅通过增加头层计算就能达到同样的效果。
先针对greedy方式进行分析，简化问题。
首先，在首次推理中，可以得到各个头的预测，但无法确认Medusa Head预测的token是否正确
暂时无法在飞书文档外展示此内容

然后，在修正过程中，包含三个步骤
1. 借助past_key_value, 可以将第一次预测的5个token当成query与20个KV进行前向计算，经过softmax得到5个next token【绿色】
2. 将5个next token与query错位校验，就能得出接受的token，如下图右上角得到accept_length为2的token。
a. 事实上仅有“语”匹配上了，X[0] 大 -> y[0] 语 = x[1] 语
b. 基于 大语 预测出来的 言 是可以被接受的，不需要再错位校验。
3. 最精髓的是Medusa Head会取accept length位置的token，当成是下一轮的输入
至此我们经过两轮forward 计算就得到了3个token, 那么加速为 3/2=1.5x, 综合步骤2和3，Medusa把verify和Next-Next-Token 一起做了，这是与Speculative Decoding 比较大的一个区别
暂时无法在飞书文档外展示此内容

至此，我们已经剖析将 Medusa 的baseline的核心实现：
1. 新增的Medusa Head需要训练, 数据构造需要偏移>1位
2. Medusa 的推理流程可以理解：Prefill + Verify + Verify + ...
3. 这里的加速比是1.5x，我们接下来思考更精妙的优化技巧，使得并行解码的接受率更高

Tree attention机制
在Medusa中，上述的Medusa Head基础版本解码采用greedy方式取Top-1 Token，实际上可以在每个Medusa Head采用Top-k 得到多个候选token，构建出树状结构，LM_head 的输出作为根节点，树的深度自顶向下遍历称为解码路径(论文里用candidates path)。
图示右图存在6条解码路径，
[It, is],[It, '],[It, the],[I, is],[I, '],[I, the],
[图片]
[图片]

LM-head 取Top-1， Medusa-head 取Top-k个tokens，假设共有4个头
- Top-3: 候选路径有81条
- Top-10: 候选路径有10000条
可想, 解码路径会随着Top-k 和头数增多急剧增加，那么新的问题是
1. 如何能减少候选解码路径？
2. 如何能在候选解码路径中，得到最优解码路径？

当Top-k 变大时，会产生大量的候选路径，具有庞大的搜索空间, 那么可以试着构造一种稀疏的树结构，能极大减少树搜索规模
- Medusa在论文里举例了一种Top-10的稀疏树结构，是手动设计的。
  - 树的层数是Medusa对应的头数-1，除掉Root
  - 每一层的token的数量为Medusa的Top-k取值
  - 手工设计的稀疏树结构，越靠左的节点，有更多的子节点路径，这样就较为合理的剪枝
  - 这样就将10000个路径的树优化到只有42条路径(树叶子结点)
  - 这里的路径可以提前结束，不要求一定要遍历到最后一层
[图片]

查看代码里的手工路径表，以上红线的路径为[0,1,1], 并且这条路线的长度可以不为4
[0,1,1] 第0个元素值为0，代表第1个medusa头top-k的token中取第0个token

Tree attention算法过程
当引入树搜索结构后，需要重新组织generate代码结构，核心包含6个步骤
1. generate_medusa_buffers : 根据设定的medusa choices得到稀疏的树结构表达
2. initialize_medusa: 首次预填充past_key_value, 便于后续verify
3. generate_candidates : 生成候选路径
4. tree_decoding : 基于tree mask每行计算logits
5. evaluate_posterior: 评估每条路径合理性
6. update_inference_inputs: 更新kv cache，preds
到此，已经了解了medusa算法的解码流程，下文会简要介绍下medusa是如何从候选集合中得到最优的候选序列

Typical Acceptance
在Medusa论文里采用了Typical Acceptance 解码策略，这种解码策略主要ref Truncation Sampling
直观理解是我们在LLM解码过程，不需要太能predictable的词，也不能有太surprising的词，这样就能保证我们能得到丰富且避免重复生成的词汇。
truncation sampling有两种实现：
- 一种是 $$\epsilon$$−sampling , 将概率低于阈值的token筛除掉
- 一种是$$\eta$$- sampling,   依赖一个固定的阈值和条件熵的阈值来自适应决定保留哪些token
Typical Acceptance：
Medusa 通过比较目标预测的概率与traction sampling 阈值来确认是否接受:
[图片]
算法流程
1. 计算熵
posterior_entropy = -torch.sum(posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1)
2. 计算阈值
threshold = torch.minimum(torch.ones_like(posterior_entropy) * posterior_threshold,torch.exp(-posterior_entropy) * posterior_alpha,)
3. 和阈值比较，计算被接受的token序列编号
posterior_mask = candidates_prob > threshold
candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
4. 获取最优的候选序列
# 获取最长的token序列长度
accept_length = candidates_accept_length.max()
# 有多个候选集有同样的长度，取likelihood最大的序列作为最优候选序列
best_candidates = torch.where(candidates_accept_length == accept_length)[0]
likelihood = torch.sum(torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1)
best_candidate = best_candidates[torch.argmax(likelihood)]

实验
trt-llm-0.9.0版本，模型vicuna-7b，medusa解码配置：temperature=0，num_medusa_heads=4
[图片]
目前的结论：
- prefill阶段变化不大，慢一点的原因是，多了medusa head计算部分
- 从推理次数加速比的角度看是符合预期（官方代码中也采用这种评估方式）
- 实际的解码速度加速比是1.3x
TODO：需要分析实际解码速度不及预期的原因
