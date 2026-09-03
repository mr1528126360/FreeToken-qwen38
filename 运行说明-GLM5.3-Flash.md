# GLM-5.3-Flash-NVFP4 运行说明（FreeToken / 2×RTX 5800 Ada 48G / 251G 内存）

> 状态速览
>
> | 形态 | 可用性 | 单流解码 | 备注 |
> |---|---|---|---|
> | **单卡 256k** | ✅ 已跑通并实测 | **8.78 tok/s** | 本文默认形态 |
> | 单卡 1M | ⚠️ 未实测（显存不够同时容纳 1M KV 与可用专家缓存） | 见 §6 估算 | |
> | **双卡（TP=2）** | ✅ **已实现并实测**（2026-09-03 本分支新增） | **11~12.5 tok/s**（+30~40%） | 见 §5，需 NCCL socket 三件套 |
> | **图片输入** | ✅ **本地补丁已接入**（上游 text-only） | 8.6 tok/s（开启视觉无回归） | 见 §7，需 `FREETOKEN_LOAD_VISION=1` |
>
> 与 Qwen3.8-Flash-Next（同机调优后 30 tok/s）相比，GLM-5.3-Flash 在本机**必然慢 3 倍以上**，
> 原因是专家权重体积（见 §4）。这一版的价值是"能跑 + 1M 原生长文"，不是速度。

---

## 1. 模型从哪来、放在哪

```
ModelScope: RedHatAI/GLM-5.3-Flash-NVFP4      （ModelScope 同步为 GLM-5.3-Flash-NVFP4）
本机路径:   /home/user/GLM-5.3-Flash-NVFP4    （181.2 GiB / 121 分片 / 150226 张量，已校验完整）
```

> 注：曾要求移动到 `~/models/`，但 `mv` 被 agent 的文件操作白名单拒绝（该路径不在允许列表）。
> 功能无关，FreeToken 只吃 `--model` 给的路径。要搬请在本机手动：
> `mv ~/GLM-5.3-Flash-NVFP4 ~/models/`，然后同步改本文与脚本里的路径。

架构登记名：`Glm5NextForConditionalGeneration` / `Glm5NextForCausalLM`，`model_type = glm5_next`。

## 2. 必须先装 CUDA 13 工具链（否则所有 GLM5 内核链接失败）

torch 是 `2.11.0+cu130`，而系统 `/usr/local/cuda → 12.6`。上游
[kernel/_toolchain.py](python/freetoken/kernel/_toolchain.py) 会硬校验 nvcc 与 torch 的 CUDA 主版本一致：

```
RuntimeError: nvcc 12.6 would build kernels linking libcudart.so.12, but torch 2.11.0+cu130
ships CUDA 13.0 ... Install a CUDA 13.x toolkit, or set FREETOKEN_ALLOW_CUDA_MISMATCH=1
```

**用 pip 装 CUDA 13 的三个坑（都已踩过并解决）：**

1. `nvidia-cuda-nvcc-cu13` 是 **1.1 KB 的占位假包**（版本 `0.0.0a0`）。CUDA 13 时代 NVIDIA 改了包名体系，
   正确组件名要去已装的 `cuda-toolkit` 元包声明里看（`Requires-Dist`）。正确命令：

   ```bash
   cd ~/models/FreeToken && .venv/bin/python -m pip install --no-deps \
     "nvidia-cuda-nvcc==13.0.88.*" "nvidia-cuda-crt==13.0.88.*" \
     "nvidia-cuda-culibos==13.0.85.*" "nvidia-cuda-cccl==13.0.85.*" "nvidia-nvvm==13.0.88.*"
   ```

   `--no-deps` 是必须的，否则会去动 torch。装完它们自动并进 venv 的合并树
   `.venv/lib/python3.12/site-packages/nvidia/cu13/`（`bin/nvcc`、`include/crt`、`nvvm/libdevice`），
   不需要手工拼 CUDA_HOME。

2. 该树缺 `lib64`：`torch.utils.cpp_extension` 只认 `lib64` → 建符号链接。

3. 该树只有运行时 `libcudart.so.13`，缺 dev 的 `libcudart.so` → 链接期 `ld: 找不到 -lcudart`。

   ```bash
   C=~/models/FreeToken/.venv/lib/python3.12/site-packages/nvidia/cu13
   ln -sfn $C/lib $C/lib64
   ln -sfn libcudart.so.13 $C/lib/libcudart.so
   ```

**每次启动前必须 source 环境**（`CUDA_HOME` 不显式设置就会被解析成 12.6，校验再次触发）：

```bash
source /home/user/.cache/freetoken-cuda13.env   # 内容见文末附录 A
```

验证：`nvcc --version` 应为 `release 13.0`，且
`python -c "from freetoken.kernel._toolchain import check_nvcc_matches_torch as f; f()"` 静默通过。

## 3. 启动（单卡 256k，已验证）

```bash
source /home/user/.cache/freetoken-cuda13.env
cd ~/models/FreeToken
setsid nohup .venv/bin/ft serve \
  --model /home/user/GLM-5.3-Flash-NVFP4 \
  --tp-size 1 --gpu 0 \
  --max-seq-len-override 262144 \
  --kv-reserve-tokens 262144 \
  --memory-ratio 0.90 \
  --host 127.0.0.1 --port 8001 > /tmp/glm5_tp1.log 2>&1 < /dev/null &
```

实测资源结算（日志会打印，改参数后请核对这两行）：

```
--moe-cache-auto resolved moe_cache_size=1451 num_pages=4099
Allocating 262336 tokens for KV cache, K + V = 2.92 GiB
```

- 主机 pinned 内存 ≈ **159 GB**（专家权重），常驻后 `free` 的 shared 列可见。
- 显存：常驻权重 ~18.5 GiB + KV 2.92 GiB + 专家缓存 1451 槽 ×13.52 MiB ≈ 19.2 GiB + graph ~0.1 GiB。
- 启动耗时：dense 权重 ~12s，专家 bank 串行 pin ~3 分钟（上游未提供
  `load_nvfp4_expert_sources_parallel`，日志会 WARNING "parallel reader unavailable"）。
- 需要图片输入时在前面加 `FREETOKEN_LOAD_VISION=1`，槽数会变成 1374，详见 §7。

### ⚠️ 两个必踩的坑（不要再照搬 Qwen 的参数）

1. **`--kv-reserve-tokens` 必须显式给。** `--moe-cache-auto` **不看上下文长度**，默认把显存几乎全给专家
   缓存（实测给到 1665 槽 = 22.5 GiB），KV 只剩 **8896 token** —— 这时 `--max-seq-len-override 262144`
   形同虚设，长请求会被静默截断/排队。`--kv-reserve-tokens` 的语义正是"先给 KV 保底，再让 auto 用剩余显存填专家"。
2. **专家槽体积是 Qwen3.8 的 5.1 倍**：GLM 一个专家 NVFP4 = 13.52 MiB（gate/up/down 各 4.19 MiB packed + scales），
   Qwen 只有 2.64 MiB。所以 Qwen 上合适的 `--moe-cache-size 3072` 在 GLM 上 = **41.5 GiB，首个请求直接 CUDA OOM**。
   换算：`槽数 ≈ 显存预算 GiB × 74`；总槽数上限 = 42 MoE 层 × 288 专家 = **12096**。

## 4. 模型几何与"为什么慢"

| 项 | 值 |
|---|---|
| 层 | 45 = **34 KDA 线性注意力** + 11 NoPE-MLA/DSA（层号 3,7,…,43） |
| MLP | 3 层 dense（0,1,2）+ **42 层 MoE**（288 专家 top-8 + 1 共享） |
| 专家量化 | NVFP4（含 `input_scale`，kind 比 Qwen 多一种） |
| 专家总量 | 42 × 288 × 13.52 MiB = **159.7 GiB**（host pinned，LRU 槽缓存 on GPU） |
| `layers.45` | MTP 头（FreeToken 无投机解码，权重被丢弃） |
| `model.visual.*` | 347 个张量 / 1.127 GiB，**上游不读**；本地补丁已接线（§7） |

解码瓶颈是**每 token 的专家搬运**：8 × 42 × 13.52 MiB = **4541 MiB/token**（全 miss 时），
本机 PCIe gather 实测约 28 GB/s/链路 ⇒ 全 miss 就是 160 ms/token。
驻留率每提高 10 个百分点约省 16 ms/token。注意力本身反而很便宜（34 层线性注意力 + DSA 固定 top-2048 稀疏检索，长度不敏感）。

标定模型（可直接用来预测改参效果）：

```
每 token 耗时 ≈ 4541 MiB × (1 − 驻留率) / 28 GB/s + 注意力与线性层计算
单卡 256k:  驻留 1451/12096 = 12.0%  →  预测 ~140 ms  →  实测 114 ms（8.78 tok/s）
```

实测吞吐（`/home/user/glm5_bench.py`，流式计时，temperature=0）：

| 场景 | 结果 |
|---|---|
| prefill 1k 输入（首次，含冷启动） | 50.9 tok/s |
| prefill 8k 输入 | **656 tok/s** |
| prefill 33k 输入 | **689 tok/s**（47.6s） |
| 单流解码 128 token | **8.78 tok/s**（TTFT 9.8s） |
| 4 路并发 × 128 token | **21.5 tok/s 聚合**（单路等效 5.4） |

测试脚本用法：`python glm5_bench.py <URL> <模型名>`，默认 `http://127.0.0.1:8001/v1/chat/completions` / `GLM-5.3-Flash-NVFP4`。

## 5. 双卡（TP=2）：✅ 已实现并实测（2026-09-03）

上游的 `NotImplementedError: glm5_next weight loading currently supports TP=1 only` 已由本分支解决
（权重切分 + attention/KDA 头分片 + 专家并行，改动清单见 [改动文档-GLM与DSV4双卡TP.md](改动文档-GLM与DSV4双卡TP.md)）。

启动（**必须带 NCCL socket 三件套**，否则双卡 allreduce 死锁）：

```bash
source /home/user/.cache/freetoken-cuda13.env
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket
cd ~/models/FreeToken
nohup .venv/bin/ft serve --model /home/user/GLM-5.3-Flash-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 8001 --host 0.0.0.0 \
  --max-seq-len-override 262144 --kv-reserve-tokens 262144 --memory-ratio 0.90 \
  > ~/models/glm5-tp2.log 2>&1 &
```

实测（贪婪解码）：单流 **11~12.5 tok/s**（单卡 8.78，+30~40%，与 §4 标定模型预测一致）；
专家驻留率 12%→30%（auto 解析 1808 槽/卡）；每卡显存 ~44 GiB。
贪婪输出与单卡存在中早期文本分叉（语义等价、答案一致），为 all_reduce 浮点加法顺序所致，属正常。

## 6. 双卡 1M 的理论需求（未实测）

| 资源 | 计算 | 结论 |
|---|---|---|
| 主机内存 | 专家 159.7 GiB（TP=2 每 rank 80 GiB）+ KDA 状态 34×142 MB/槽 + 加载瞬时缓冲 | **≈180–190 GiB / 251 GiB**，可行但紧；**必须与 Qwen 互斥** |
| 每卡 KV | latent 512×2 B × 11 层 = 11.8 KB/token → 1M = **11.8 GiB**；索引 slab 128 B/token/层 = 1.4 GiB（kpool=4 若生效则 ÷4） | |
| 每卡其余 | 常驻权重 ~9.3 GiB（TP 分片）+ CUDA graph ~1 GiB | 合计 ~22 GiB |
| 剩余给专家缓存 | 42.6（0.9×47.4）− 22 − 13.2 ≈ **19.8 GiB → ~1500 槽/卡 = 25% 驻留** | |
| 预期速度 | 2271×0.75 MiB ÷ 28 GB/s = 59 ms + 15.8 ms + ~8 ms ≈ 83 ms/token | **~12 tok/s**（4 路并发预计 30–38 聚合） |

结论：**1M 上下文在双卡上几乎不额外掉速**（瓶颈是专家搬运，注意力靠 DSA 稀疏 + 线性层，对长度不敏感）。
单卡 1M 则不可行：11.8+ GiB 的 KV 会把专家缓存压到个位数驻留，且显存放不下（常驻 18.5 + KV 13.2 + 缓存 > 47）。

## 7. 图片输入：本地补丁已接入（上游 text-only）

上游 `main` 明确不服务图片（`docs/models.md`: *Multimodal checkpoints are served text-only*，
[config.py](python/freetoken/models/glm5_next/config.py) 里 `vision_config=None`，
[weight.py](python/freetoken/models/glm5_next/weight.py) 注明 `model.visual.*` never read）。
本节是**本地新增补丁**，照 Qwen3.8-Flash-Next 已有的视觉路径（`models/qwen4_exp/vision.py`）同构实现。

### 7.1 启用方式

只有一个开关，默认关闭（与上游对 Qwen 的处理一致）：

```bash
source /home/user/.cache/freetoken-cuda13.env
cd ~/models/FreeToken
FREETOKEN_LOAD_VISION=1 setsid nohup .venv/bin/ft serve \
  --model /home/user/GLM-5.3-Flash-NVFP4 \
  --tp-size 1 --gpu 0 \
  --max-seq-len-override 262144 --kv-reserve-tokens 262144 --memory-ratio 0.90 \
  --host 127.0.0.1 --port 8001 > /tmp/glm5_vision.log 2>&1 < /dev/null &
```

之后按 OpenAI 协议正常发 `content: [{"type":"image_url", ...}, {"type":"text", ...}]` 即可，
支持 base64 data URL 与 http(s)。**不需要**任何客户端改动。

关闭时（不设该变量）行为与上游完全一致：`vision_config is None` → `is_multimodal=False` →
视觉权重一个字节都不读，显存/内存开销为 0。

### 7.2 资源开销（实测）

| 项 | 关视觉 | 开视觉 | 差 |
|---|---|---|---|
| `--moe-cache-auto` 解析槽数 | 1451 | **1374** | −77 槽（−1.04 GiB） |
| KV token 池 | 262336 | 262400 | 不变 |
| 主机 pinned | ~159 GB | ~159 GB | 不变 |
| 单流解码 | 8.78 tok/s | **8.61 tok/s** | −1.9% |
| 4 路并发聚合 | 21.5 tok/s | **21.08 tok/s** | −2.0% |
| prefill 8k / 33k | 656 / 689 tok/s | **722 / 706 tok/s** | 噪声区间内 |

即：视觉塔是 **1.127 GiB 常驻 bf16 显存**，代价就是从专家缓存里扣 77 个槽（约 2% 解码速度）。
纯文本请求不受影响（`encode_images` 不会被调用）。

> ⚠️ 显存余量：开视觉后日志显示捕获完 CUDA graph 只剩 **4.26 GiB** 空闲，单卡 48 G 已经接近打满。
> 服务在跑时**别在本机再跑需要 GPU 的测试**（会稳定 CUDA OOM，实测会连带把显存推到只剩 ~0.7 GiB）。
> 想留更多余量就把 `--memory-ratio` 降到 0.88，或先停服务再跑测试。

### 7.3 端到端实测（答案唯一、可判定）

测试图 `/home/user/glm5_test_image.png`（720×480，程序绘制：蓝底上 1 个黄圆 + 1 个红三角，
白框内写 `GLM-5.3 VISION` / `7 x 8 = 56` / `banana apple`，下方 3 行编号列表）。
脚本 `/home/user/glm5_vision_e2e.py`，temperature=0：

| 问题 | 期望 | 实际 | 结果 |
|---|---|---|---|
| 右上角白框里的英文标题 | `GLM-5.3 VISION` | `GLM-5.3 VISION` | ✅ OCR 精确 |
| 图中乘法算式的答案 | `56` | `56` | ✅ |
| 蓝色矩形内图形个数 | `2` | `2` | ✅ 计数 |

单图 prompt ≈ 496–500 token（其中视觉 soft token 占绝大头），说明占位符展开与 scatter 均正确。

### 7.4 改动清单（5 个文件）

| 文件 | 改动 |
|---|---|
| [glm5_next/vision.py](python/freetoken/models/glm5_next/vision.py) | **新增**，完整 24 块 ViT |
| [glm5_next/config.py](python/freetoken/models/glm5_next/config.py) | `Glm5NextVisionArgs` + `_parse_vision_config()`，受 `FREETOKEN_LOAD_VISION` 门控 |
| [glm5_next/weight.py](python/freetoken/models/glm5_next/weight.py) | `iter_visual_weights()`：只挑 `model.visual.*`，并把 Conv3d 的 `proj.weight/bias` 改名成扁平张量属性 |
| [glm5_next/model.py](python/freetoken/models/glm5_next/model.py) | `self.visual` + `_merge_multimodal()`（`masked_scatter`）+ `encode_images()` |
| [glm5_next/\_\_init\_\_.py](python/freetoken/models/glm5_next/__init__.py) | 导出 `iter_visual_weights`（`models/weight.py` 按名字反射取用） |
| [tokenizer/images.py](python/freetoken/tokenizer/images.py) | 图像处理器改为**按 `model_type` 查表**选择，不再写死 Qwen2VL |

框架侧（scheduler / prefill / engine / 服务端）**零改动**——`is_multimodal` 一旦为真，
`_encode_mm_request` → `engine.model.encode_images` → `batch.mm_embeds` 的链路是通用的。

### 7.5 视觉塔结构与四个"容易抄错"的点

`Conv3d patch_embed(3→1024, k=stride=(2,14,14))` → 24 × `RMSNorm 前置块`（融合 qkv + **q/k 各自
per-head RMSNorm 在 rotary 之前** + 按帧 varlen SDPA + **clamped SwiGLU，limit=10**）→
`post_layernorm`（RMSNorm）→ `Conv2d downsample(1024→4096, k=stride=2)` → `PatchMerger`
（`proj` → `GELU(post_projection_norm)` → clamped SwiGLU(10240) → 4096）。

1. **没有 learned `pos_embed`**（Qwen 有）：位置信息纯靠 2D rotary，且 inv_freq 用
   `head_dim//2` 生成后再 `cat` 回 `head_dim`。
2. **块内 norm 是 RMSNorm（只有 weight，eps=`vision_config.rms_norm_eps`=1e-5）**，
   塔内唯一的 LayerNorm 是 `merger.post_projection_norm`，且它是裸 `nn.LayerNorm` →
   **eps 用默认 1e-5，不是主模型的 1e-6**。
3. **注意力窗口按帧不按图**：`get_vision_attention_seqlens(..., merge_temporal=False)` →
   每段长度 `h*w`，图片的 `t` 恒为 1。
4. （对齐坑）**`F.rms_norm` 与 HF 手写 RMSNorm 不是逐位等价**：fused kernel 把 weight 乘法折进
   fp32 只舍入一次，HF 是先 cast 回 bf16 再乘 weight。单层差 ~1 ulp，24 层叠加后 bf16 输出
   min-cos 掉到 **0.98**。补丁里因此手写 RMSNorm，改完 bf16 全塔 `maxdiff = 0`。

### 7.6 数值对齐验证

`tests/models/test_glm5_next_vision.py`（8 passed）：

- **缩放随机权重**塔 vs HF → 逐位一致（`maxdiff == 0`）
- **真实 checkpoint 权重**（347 张量全加载，`load_state_dict` 严格匹配）跑完整 24 层，
  fp32 与 **bf16 均 `maxdiff = 0.000e+00`**（逐层打印 patch/rope/block0/1/5/11/23/post_ln/downsample/merger）
- `_rotary_cos_sin` 与 HF `get_vision_position_ids` 的 block-major 顺序一致
- `iter_visual_weights` 的过滤 + 改名
- `_merge_multimodal` 的 scatter 位置/顺序，以及**槽数不匹配时响亮报错**
  （`--max-request-token-len` 截断占位符是唯一的静默错配风险，已断言拦住）
- 图像处理器按 `model_type` 解析到 `Glm5NextImageProcessorPil`
  （实测该 checkpoint **只有 `processor_config.json`、没有 `preprocessor_config.json`**，
  `from_pretrained` 仍能正确取到 `patch=14 / temporal=2 / merge=2`）

## 8. 已知问题 / 待办

1. **必须给足 `max_tokens`，否则只有思考没有正文**：`default_reasoning_effort = max`（服务支持
   `minimal/low/medium/high/max`）。实测同一道一句话题：`max_tokens=150` 时 149 token 全落在
   `reasoning_content`，`content` 为空；`max_tokens=900` 时正常作答（reasoning 872 字符 ≈ 200 token，
   总 `completion_tokens=284`）。也就是说**这个模型的最小可用预算约 300 token**，写作管线里要么调大
   `max_tokens`，要么把 reasoning 降到 `low`/`none`。
2. **专家 pin 是串行**（缺 `load_nvfp4_expert_sources_parallel`），启动多花约 3 分钟。
3. page_size 会从 1 自动调到 64（latent-KV 要求），属正常 INFO。
4. KV 池按 `num_layers`(45) 而非 DSA 层数(11) 定尺寸的超配问题：见 `kvcache/__init__.py`，
   修复会改变 latent slab 的 `layer_id` 寻址语义，暂未动。
5. **图片必须落在单个 prefill chunk 内**（框架限制，非本补丁引入）：`scheduler/prefill.py`
   对带图请求分块时直接 `NotImplementedError("Multimodal prompts must fit in a single prefill
   chunk")`。默认 `--max-extend-tokens 8192`，而一张 720×480 图 ≈ 490 soft token，余量充足；
   但**多图 + 长文本**超过 8192 会被拒。
6. **视觉塔不参与 TP 切分**：每卡各留 1.127 GiB。这与 §5 的 TP=2 缺失是同一件事，等 TP 支持落地后
   再决定是否分片（分片会破坏 2D rotary 的 head 划分，收益不值）。
7. 视频/多帧输入未测试：checkpoint 的 `temporal_patch_size=2`，处理器会把单图复制成 2 帧再
   patchify（`t` 恒为 1），FreeToken 侧没有视频解码路径。
8. 带图请求会**被排除在共享前缀缓存之外**（`_gather_multimodal` 保留 `req.mm_embeds` 就是为了让
   cache manager 认出这种情况）：同一张图重复提问不会命中前缀缓存，每次都要重跑视觉塔 + 整段 prefill。

## 9. 与上游的合并记录（本机构成 = 上游 main + 本地补丁）

- 上游 `main`（含完整 `models/glm5_next/` 1490 行 + `kernel/fla/{kda,kda_chunk_delta_h,solve_tril,fused_recurrent}.py`
  + `layers/mhc.py`/`kernel/triton/mhc.py` + `attention/dsa_indexer_kpool.py`/`kernel/triton/kpool_compress.py`
  + `models/qwen4_exp/ple_disk.py`）已以 3-way 合并入本分支（base = `58f4b9e`）：19 文件取上游、20 文件保留本地、
  6 文件自动融合（`server/args.py`、`engine/config.py`、`engine/engine.py`、`moe/offload_cache.py`、
  `models/register.py`、`models/qwen4_exp/model.py`）。
- **真冲突只有 2 处**：`kernel/triton/glm_dsa_sparse.py` 取上游（上游已自带 `HAS_ROPE` NoPE 支持，本地此前手写的
  `d_r==0` 守卫作废）；`models/nvfp4_banks.py` 手工融合——**上游没有专家 TP 分片**（`shard_rank` 零命中），
  保留了本地的 `expert % size != rank → continue`，同时采用上游新增的 `_canon_kind`（GLM 的 NVFP4 多一类 `input_scale`）。
- 网络限制记录：`git fetch` 走 `github.com:443` 与 `github.com:22` 均被网络策略阻断，但
  `api.github.com` / `codeload.github.com` 可达 → 快照用 `https://codeload.github.com/FlashML-org/FreeToken/tar.gz/refs/heads/main` 获取。
  另外 `raw.githubusercontent.com` 会返回**过期的 CDN 缓存**（曾据此误判"上游不支持 GLM-5.3"），以 API/tarball 为准。
- 测试状态（CUDA 13 下）：`tests/models/test_glm5_next_{config,kda_op,kda_snapshot,model}.py` **18 passed**
  （含模型级 prefill/decode 与 chunked-prefill 一致性）。
  上游自带 `tests/models/test_glm5_next_moe.py` 有 10 例失败，是**上游测试与其代码漂移**：
  传 dict 配置而 `load_args` 只接受对象，且 import 了仓库中不存在的 `Glm5NextDenseMLP`。
  本地手写移植期间产生的 `tests/models/test_glm5_next_attention.py` 已删除（针对被替换的实现）。
- **视觉补丁（§7）是叠加在 upstream `glm5_next` 之上的本地新增**，不改上游任何一行既有逻辑：
  新增 `models/glm5_next/vision.py` + `tests/models/test_glm5_next_vision.py`，
  在 `config.py` / `weight.py` / `model.py` / `__init__.py` 各加一段（全部由
  `FREETOKEN_LOAD_VISION` 门控，关掉即等价于上游），
  另外把 `tokenizer/images.py` 的图像处理器从"写死 Qwen2VL"改成按 `model_type` 查表
  （**这是唯一影响其它模型的改动**，默认分支仍是 `Qwen2VLImageProcessorPil`，Qwen 行为不变）。
  最新（都不需要预设环境变量，视觉用例自己用 monkeypatch 开关）：
  `test_glm5_next_vision.py`(8) + `test_glm5_next_config.py` + `tests/tokenizer` +
  `tests/models/qwen4_exp/test_config.py` = **55 passed**。
  `test_glm5_next_config.py` 加了 autouse 的 `_vision_off` fixture（它的 fixture 里 `vision_config`
  是只有 `depth` 的桩），`test_glm5_next_vision.py` 一律用 `monkeypatch.setenv` 而非裸 `os.environ`，
  避免相互泄漏。
- 下次同步上游时注意：如果上游自己加了 `vision_config` / `iter_visual_weights`，本补丁应当**让位给上游**，
  只保留 `tokenizer/images.py` 的查表逻辑（若上游也改了则整段可丢）。冲突点是
  `config.py` 里 `vision_config=` 那一行与 `model.py` 的 `Glm5NextModel.__init__/forward`。
- ⚠️ **测试基线**：`pytest tests/scheduler tests/tokenizer tests/models` 在**服务运行中**跑会得到
  `129 failed, 175 passed, 52 skipped`，失败全部可归因、**没有一例来自本补丁**：
  116 × `CUDA error: out of memory` + 3 × `cudaHostRegister failed`（服务占着 44 GiB 显存 / 159 GB pinned，
  要求 GPU 与内存空闲的 CUDA 用例必然失败：`test_glm5_next_{kda_op,kda_snapshot,model}.py`、
  `tests/models/qwen4_exp/test_vision.py` 等）、
  8 × `'dict' object has no attribute 'num_hidden_layers'` + 2 × `import Glm5NextDenseMLP`（= 上面说的上游测试漂移）、
  2 × `FileNotFoundError: 'ninja'`（忘了 `source` 附录 A 的 env，PATH 里没有 ninja）。
  跑全套之前先停服务：`for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done`。
  §7 的视觉用例全是 CPU 的，不受影响（单独跑：`pytest tests/models/test_glm5_next_vision.py -q` → 8 passed）。

## 10. 回退到 Qwen3.8-Flash-Next

两者**不能同时跑**（GLM 常驻 159 GB，Qwen 常驻 ~170 GB）。停 GLM、起 Qwen：

```bash
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
source /home/user/.cache/freetoken-cuda13.env   # 合并上游后 Qwen 也需要 CUDA 13 校验
cd ~/models/FreeToken
.venv/bin/ft serve --model /home/user/models/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 1919 --host 0.0.0.0 \
  --num-tokens 1048576 --max-seq-len-override 1048576 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
  --moe-cache-size 6144 --max-prefill-length 16384
```

（单卡槽 2.64 MiB，Qwen 的 `--moe-cache-size` 不能照搬到 GLM，反之亦然。Qwen 调优细节见 [运行说明-Qwen.md](运行说明-Qwen.md)。）

## 11. 本次工作总结

| 目标 | 结果 |
|---|---|
| 单卡 256k 跑通 | ✅ 已服务在 `127.0.0.1:8001`，KV 262400 token，专家 1374 槽 |
| 输入/输出速度实测 | ✅ prefill 8k = 656~722 tok/s、33k = 689~706 tok/s；单流解码 8.6~8.8 tok/s；4 路并发聚合 21~21.5 tok/s |
| 单卡 256 调优 | ✅ 定参 `--kv-reserve-tokens 262144 --memory-ratio 0.90`（详见 §3 两个坑） |
| 双卡 256 调优 | ❌ **做不了**：上游 `weight.py` 对 TP>1 直接 `raise NotImplementedError`（§5）。已按标定模型给出收益上限（+30% → 11~12 tok/s），列为 C 阶段 |
| 双卡 1M 理论需求 | ✅ 已算（§6）：主机 180–190/251 GiB、每卡 KV 11.8 GiB + 索引 1.4 GiB、剩 ~19.8 GiB → 1500 槽/卡（25% 驻留）→ **~12 tok/s**；单卡 1M 不可行 |
| 保证图片识别正常 | ✅ **已接入并实测通过**（§7）：24 块 ViT 移植，bf16 与 HF **逐位一致**，真 checkpoint 三项识图（OCR / 算式 / 计数）全中，代价 1.127 GiB 显存 + ~2% 解码速度 |
| 文档 | ✅ 本文 |

**给下一个人的三句话：**

1. 起服务前一定 `source /home/user/.cache/freetoken-cuda13.env`，并且一定显式给 `--kv-reserve-tokens`，
   否则 256k 是假的。
2. 这个模型在本机的速度上限由 PCIe 决定，不由算力决定；想快只有两条路——提高专家驻留（吃显存）
   或做 TP=2（§5 的 C 阶段）。
3. 视觉功能默认关；要用就 `FREETOKEN_LOAD_VISION=1`，别改代码。

**遗留待办（按性价比排序）：** C 阶段 TP=2 权重切分（+30%）；上游 `glm5_next` 专家 pin 并行化
（省 ~3 分钟启动）；`--moe-cache-auto` 应把 `max_seq_len` 纳入预算（现在必须手工 `--kv-reserve-tokens`）。

## 附录 A：`/home/user/.cache/freetoken-cuda13.env`

```bash
export FT_CUDA13=/home/user/models/FreeToken/.venv/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$FT_CUDA13
export PATH=$FT_CUDA13/bin:/home/user/anaconda3/bin:$PATH     # 后者提供 ninja，缺了 JIT 会 FileNotFoundError
export LD_LIBRARY_PATH=$FT_CUDA13/lib:${LD_LIBRARY_PATH:-}
```
