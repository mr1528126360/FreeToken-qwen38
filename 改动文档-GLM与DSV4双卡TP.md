# 改动文档：GLM-5.3-Flash 与 DeepSeek-V4-Flash 双卡 TP=2 支持

> 2026-09-03，在 `feat/qwen4-exp-tp-vision` 分支上新增。
> 沿用 qwen4_exp TP 的既有框架（pynccl 通信、`layers/linear.py`/`layers/embedding.py` 原语、
> `layers/moe.py` 专家并行），本次只接两个新模型，**通用层零改动**。

## 策略

| 模型 | attention | MoE 路由专家 | embedding/lm_head | 共享专家 |
|---|---|---|---|---|
| GLM-5.3（dense 为 bf16） | **完整头分片**（64 头 ÷2）+ o_proj 行分片 all_reduce | 专家并行（`id % 2 == rank`） | 词表 div_ceil 切（本来就是 TP 类） | gate_up 列切 + down 行切 |
| DeepSeek-V4（dense 为 block-fp8，128×128 scale 块不可切） | **整层复制**（两卡输入一致、输出天然一致，无需 all_reduce） | 专家并行（同上） | 词表 div_ceil 切 | 复制（fp8） |

DSV4 的 attention 复制是刻意取舍：block-fp8 的 scale 块按 128 对齐，头维不对齐无法切；
专家（154GB）和词表才是内存大头，attention 复制的计算代价可忽略。

## GLM-5.3（models/glm5_next/）改动清单

| 文件 | 改动 |
|---|---|
| `weight.py` | 删除 TP=1 的 raise（原 :222-226）；新增 `_head_rows` + `_shard_dense_weight`：DSA `q_b/kv_b_proj` 按头 dim0 切、`o_proj` dim1 切；KDA `in_proj` 混合切（q\|k\|v\|b 按头、低秩 `f_a\|g_a` 复制）、`conv1d` 按 [q\|k\|v] 通道切、`A_log`/`dt_bias`/`f_b`/`g_b` 按头切；embed/lm_head 词表切；其余（latent、indexer、hc、norm、router）复制；`load_nvfp4_expert_sources` 传 `expert_shard=(rank,size)` |
| `attention.py`（DSA/MLA） | 头数 `div_even`（不可整除 raise）；bf16 下 `q_b/kv_b_proj`→`LinearColParallelMerged`、`o_proj`→`LinearOProj`；latent/norm/indexer 复制；fp8 attn + TP>1 raise |
| `kda.py` | 新增 `_KDAFusedInProj`（混合布局分片，tp=1 时与原键名/数学完全一致）；头数/conv_dim 本地化；`f_b/g_b_proj`→`LinearColParallelMerged`（输出按头）；bf16 `o_proj`→`LinearRowParallel`；fp8 模式 TP=1-only |
| `mlp.py`（共享专家） | bf16 下 gate/up→`LinearColParallelMerged`、down→`LinearRowParallel`；fp8 模式 TP=1-only |
| `model.py` | `Glm5Fp8LMHead` TP>1 raise（权重全词表且无 all_gather） |
| `attention/dsa.py`（共享层，唯一例外） | 后端 `num_heads` 改用 `div_even(config.num_qo_heads, tp_size)`——层传入的 q 已是本地头数；TP=1 恒等 |

关键布局结论（读 checkpoint 头验证）：KDA 的 `f_a/g_a_proj` 是 [128,4096] **低秩共享、非按头**，
`f_b/g_b` 消费完整 128 维，因此 f_a/g_a 只能复制、f_b/g_b 按头列切——这是唯一自洽切法。

## DeepSeek-V4（models/deepseek_v4/）改动清单

| 文件 | 改动 |
|---|---|
| `model.py` | 裸 `nn.Embedding`→`VocabParallelEmbedding`、裸 head Parameter→`ParallelLMHead`（新增 `_embed_tokens`/`_head_logits` helper，key 名不变，TP=1 逐字节不变） |
| `moe.py` | **修 TP 致命 bug**：`DSV4OffloadMoELayer._prefill_routed` 的 on-demand 分支原来绕过基类 `_partition_topk` 直接用全局 id 取专家（TP 下必取错）；现补分片路由（partition → 非 owned 置 weight=0/id=-1 → clamp 到 dump row 0）；streaming/on-demand 交叉条件改用 `num_experts_global` |
| `weight.py` | `iter_weights` 只切 embed/head（词表），其余全复制；`load_dsfp4_expert_sources{,_parallel}` 与 `dummy_dsfp4_expert_sources` 加 `expert_shard`（过滤 `id%size!=rank`、bank 按 E//size 分配、本地行 `id//size`） |
| KV cache | `dsv4_paged_pool.py` 无需改动：MLA latent KV 每卡全量复制 |

## 验证

单测（GPU1，CUDA_VISIBLE_DEVICES=1）：
- `tests/models/glm5_next/`（新增 test_tp.py 等 10 例）+ `tests/models/deepseek_v4/`（新增 15 例）+ `tests/models/qwen4_exp/test_tp.py` → **37 passed**
- 相邻回归（tests/moe、tests/dsv4、tests/kvcache、tests/scheduler 相关）→ **108 passed**

真机双卡（2×RTX 5880 Ada，NCCL socket 三件套必须：`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket`）：

| 模型 | 单卡 decode | TP=2 decode | 提升 | 贪婪一致性 |
|---|---|---|---|---|
| GLM-5.3-Flash | 7.9–8.6 tok/s | **11–12.5 tok/s** | +30~40% | 4 条 prompt 均有中早期分叉，但语义等价（数值/答案一致）——all_reduce 浮点加法顺序所致 |
| DeepSeek-V4-Flash | 9.9–11.7 tok/s | **12.7–14.6 tok/s（640 槽）/ 16.5–18.8 tok/s（1200 槽）| +25~30% | 同上 |

资源（TP=2，每卡）：GLM 显存 ~44GB（auto 专家缓存 1808 槽/卡，驻留率 12%→30%）、
DSV4 显存 ~29GB（640 槽/卡）；RAM 总量不变、两进程对半分。

## 启动命令（TP=2）

```bash
source /home/user/.cache/freetoken-cuda13.env
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket   # 本机双卡跨 NUMA，必须
cd ~/models/FreeToken

# GLM-5.3-Flash
nohup .venv/bin/ft serve --model ~/GLM-5.3-Flash-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 8001 --host 0.0.0.0 \
  --max-seq-len-override 262144 --kv-reserve-tokens 262144 --memory-ratio 0.90 \
  > ~/models/glm5-tp2.log 2>&1 &

# DeepSeek-V4-Flash
nohup .venv/bin/ft serve --model ~/models/DeepSeek-V4-Flash \
  --tp-size 2 --gpu 0,1 --port 1919 --host 0.0.0.0 \
  --num-tokens 262144 --moe-cache-size 640 \
  > ~/models/dsv4-tp2.log 2>&1 &
```

注意：三个模型（Qwen/GLM/DSV4）两两不能同时跑（内存不够）；`--moe-cache-size` 在 TP 下是**每卡** slot 数；TP>1 禁止 runtime cache rebuild。

## 遗留

- DSV4 attention 若要真分片，需解决 block-fp8 按头切（头维不对齐 128 scale 块）——首阶段不做。
- GLM 的 fp8 resident 模式（attn/mlp/lm_head opt-in env）全部限制 TP=1。
- GLM 视觉塔不参与 TP（每卡各 1.127 GiB 复制，刻意保留）。
- 两卡贪婪解码会因 all_reduce 加法顺序出现中早期文本分叉（语义等价），属正常浮点行为。
