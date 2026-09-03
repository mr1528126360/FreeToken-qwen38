# FreeToken 定制改动文档：Qwen3.8-Flash-Next 双卡 TP + 1M 上下文 + 图片理解

> 本文档记录 `~/FreeToken` 仓库 `feat/qwen4-exp-tp-vision` 分支上的全部自研改动。
> 上游 FreeToken 0.1.2 原版：Qwen3.8-Flash-Next 仅支持单卡、256k 上下文、纯文本。
> 本分支新增：**双卡张量并行（TP=2）**、**1M 上下文（YaRN 扩展）**、**图片理解**。

## 一、改动总览

共 31 个文件修改（+1172/-90 行）+ 5 个新文件。全部改动未 commit，
位于分支工作区，可随时回退（见第六节）。

### 1. 双卡张量并行（TP=2）

原版引擎有 TP 框架（多 rank 启动、NCCL 通信、TP 感知的基础层），但
DeepSeek-V4 / qwen4_exp 的模型实现和 MoE offload 后端都没接 TP。本次打通：

| 文件 | 改动 |
|---|---|
| `models/qwen4_exp/attention.py` | QSA 注意力头按卡分片（24 q 头→12/卡，2 KV 头→1/卡）；`o_proj` 换 `LinearOProj`（行分片 + all_reduce）；QSA indexer 保持复制 |
| `models/qwen4_exp/gdn.py` | 线性注意力的 k/v 头按卡分片（16/48 → 8/24），`out_proj` 换行分片 |
| `models/qwen4_exp/weight.py` | dense 权重按 `shard_tensor` 分片加载（融合后按头边界切分）；移除 "TP=1 only" 限制 |
| `layers/moe.py` | 新增 `partition_topk_experts()`：MoE 专家按卡二分（`expert_id % 2 == rank`），每卡只算自己一半专家，输出 all_reduce 求和（数学等价） |
| `moe/offload_cache.py` | GPU 专家缓存槽位初始化清零（防 0×NaN）；专家槽映射支持本地子集 |
| `models/nvfp4_banks.py` | NVFP4 专家 bank 加载支持按 rank 过滤——**每卡内存从 126GB 降到 63GB** |
| `engine/engine.py` | MoE 缓存自动 sizing 按本地专家数；新增 rope scaling 覆盖逻辑（见下） |
| `kernel/pynccl.py` | 修复 pip 版 NCCL 只有 `libnccl.so.2` 导致的链接失败（否则 TP=2 根本起不来） |

正确性验证：TP=2 与 TP=1 贪婪解码对比，4 条测试 prompt 中 3 条逐字一致、
1 条仅末句存在浮点级差异（all_reduce 加法顺序所致，正常）。

### 2. 1M 上下文

模型官方上限 256k（config `max_position_embeddings=262144`）。本次新增
`--rope-scaling` 启动参数，用 YaRN（factor=4）把 rope 表扩展到 1M：

| 文件 | 改动 |
|---|---|
| `engine/config.py` | EngineConfig 新增 `rope_scaling_override` 字段 |
| `server/args.py` | 新增 `--rope-scaling` CLI 参数（JSON） |
| `engine/engine.py` | `_adjust_config` 在 rope 表长度检查前应用 YaRN 覆盖，同步更新模型级和各 attention group 的 rotary 配置 |

注意：**1M 是 off-label 扩展，官方只背书 256k**。已验证 82.7 万 token
"大海捞针"检索正确，但超长程生成质量无官方保证。

### 3. 图片理解

模型本身是多模态的（checkpoint 含 333 个 `model.visual.*` ViT 权重），
原版 FreeToken 直接丢弃视觉权重、API 收到图片不处理。本次从零补齐：

| 文件 | 改动 |
|---|---|
| `models/qwen4_exp/vision.py`（新） | 27 层 ViT 视觉编码器（conv3d patch_embed、attention block、pos_embed 插值、2D rotary、spatial merger），与 HF 参考实现**逐位一致**（cosine=1.0） |
| `models/qwen4_exp/mrope.py`（新） | mrope 三维位置编码（文本段 1D + 图片段 (t,h,w) 网格），与 HF `get_rope_index` 完全相等 |
| `tokenizer/images.py`（新） | base64 data URL / http(s) 图片解码、PIL 预处理（按 `preprocessor_config.json`）、image pad token 展开 |
| `server/generation.py`、`message/backend.py`、`tokenizer/*` | API 解析 OpenAI 多模态 content parts，图片随请求管线进入引擎 |
| `scheduler/scheduler.py` 等 | prefill 时跑 ViT 并注入 embedding（替换 image pad token 位置的 embed 输出）；无视觉能力时明确拒绝图片请求 |
| `attention/qsa_sparse.py`、`engine/graph.py` | rope 支持 per-token mrope 位置表；decode CUDA graph 兼容 |
| `models/qwen4_exp/weight.py` | `iter_visual_weights` 独立加载视觉权重（bf16，~1.3GB/卡） |

默认关闭，`FREETOKEN_LOAD_VISION=1` 开启（沿用 gemma4 的 opt-in 约定）。
限制：仅图片（无 video/audio）；带图请求不进前缀缓存、不可跨 prefill 分块。
图片 token 上限 = `--max-prefill-length`（启动命令给的是 16384，图片预处理本身
会把像素压到 ~16.3k tokens 以内，所以单边图基本都能收；图+长文本超限时返回
HTTP 400 `image prompt is too long`，不会再把服务打崩——原版这里会直接
raise 杀掉调度进程，本次已在 `scheduler/scheduler.py` 准入阶段修复为优雅拒绝）。

### 4. 顺带修复

- `kernel/pynccl.py`：NCCL 库解析（上述）
- `moe/offload_cache.py`：slot cache 清零（hybrid 路径既有 NaN 隐患）
- `tests/moe/test_nvfp4_backends.py`：tp_info fixture（既有测试隔离问题）

## 二、启动方式

所有启动都必须带修补版 CUDA 环境变量（修复系统 CUDA 13.1 与新 glibc 的
`rsqrt` 头文件冲突，否则 JIT 编译 kernel 失败）：

```bash
export CUDA_HOME=~/cuda-13.1-patched
export PATH=$CUDA_HOME/bin:$PATH
cd ~/FreeToken
```

### 方式 A：单卡（256k 上下文，日常推荐）

```bash
nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --gpu 0 --port 1919 --host 0.0.0.0 \
  --num-tokens 262144 --moe-cache-size 1024 > ~/freetoken-qwen-serve.log 2>&1 &
```

- 官方上下文上限 256k，无需 rope 扩展
- `--moe-cache-size 1024` 必须显式给（否则自动缓存吃满显存，KV 分配 OOM）
- 如需图片理解，前面加 `FREETOKEN_LOAD_VISION=1`

### 方式 B：双卡（1M 上下文）

```bash
export FREETOKEN_LOAD_VISION=1   # 可选：图片理解
nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 1919 --host 0.0.0.0 \
  --num-tokens 1048576 --max-seq-len-override 1048576 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
  --moe-cache-size 1024 --max-prefill-length 16384 > ~/freetoken-qwen-serve.log 2>&1 &
```

- `--moe-cache-size` 在 TP 下是**每卡**的 slot 数
- 加载约 5~8 分钟；就绪判断：`ft ctl health` 显示 `status=ok`

### 共用说明

- 服务地址：本机 `http://127.0.0.1:1919`，局域网 `http://<服务器IP>:1919`
  （OpenAI `/v1/chat/completions`、Anthropic `/v1/messages` 兼容，无鉴权勿暴露公网）
- **Qwen 与 DeepSeek-V4-Flash 不能同时运行**（内存合计超 247GB，会被 OOM 双杀）

## 三、关闭服务

```bash
# 1. 杀主进程
pkill -f "venv/bin/ft serve"
sleep 10

# 2. 检查显存是否释放（子进程有时残留）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 3. 若未释放，按 PID 清理残留的引擎 worker（保留桌面进程 /usr/*）
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  case "$(ps -o cmd= -p $p 2>/dev/null)" in
    /usr/*) ;;
    *) kill -9 $p 2>/dev/null ;;
  esac
done
```

## 四、双卡实测性能（TP=2，1M KV 配置，随机数字提示词无前缀缓存）

| 指标 | 双卡 TP=2 | 单卡 TP=1（256k 配置） |
|---|---|---|
| 输入 prefill（30k tokens） | **2240 tok/s** | 2126 tok/s |
| 输入 prefill（120k tokens） | **2230 tok/s** | 3119 tok/s |
| 输出 decode | **43.6 tok/s** | 40.8 tok/s |
| VRAM | 每卡 ~31GB（1M KV） | ~23GB（256k KV） |
| 内存 | 两进程合计 ~190GB | ~125GB |

结论：双卡 decode 略快（+7%），长 prefill 略慢（all_reduce 开销），
**双卡的核心价值是容量**——专家分片省一半内存、KV 分片让 1M 上下文放得下。

长上下文实测（双卡 1M 配置）：
- 24.8 万 tokens needle：命中，prefill 2125 tok/s
- 82.7 万 tokens needle：命中，prefill 1927 tok/s，用时 7.2 分钟

## 五、图片理解实测

输入测试图（红圆 + 蓝方块 + "FreeToken" 文字），模型回答完整正确识别了
形状、颜色和文字。请求格式（OpenAI 标准多模态）：

```json
{"messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
  {"type": "text", "text": "描述这张图"}
]}]}
```

端到端验证脚本：`/tmp/vision_e2e.py`（含纯文本对照、单图、多轮带图三组请求）。

## 六、回退

```bash
cd ~/FreeToken
git stash                     # 暂存全部改动（含新文件需 git stash -u）
git checkout main             # 回到官方原版
# 恢复：git checkout feat/qwen4-exp-tp-vision && git stash pop
```

回退后用单卡命令重启验证服务正常即可。原版单卡配置参考 `运行说明-Qwen.md`
的 git 历史或 `运行说明.md`（DeepSeek 版，结构相同）。

## 七、测试基线

分支上新增/更新测试：`tests/models/qwen4_exp/test_tp.py`（10 个 TP 测试）、
`test_vision.py`（11 个视觉测试）、`tests/moe/test_offload.py`（+7 个
offload TP 测试）等。全量回归（patched CUDA 环境下）：

```bash
export CUDA_HOME=~/cuda-13.1-patched PATH=$HOME/cuda-13.1-patched/bin:$PATH
cd ~/FreeToken
.venv/bin/python -m pytest tests/models/qwen4_exp/ tests/moe/ tests/models/ -x -q
# 283 passed, 55 skipped
.venv/bin/python -m pytest tests/scheduler tests/server tests/tokenizer tests/engine -q
# 758 passed
```
