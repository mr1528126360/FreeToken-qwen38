# FreeToken-Qwen3.8：双卡 TP / 1M 上下文 / 图片理解增强版

> 本项目是基于 [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
> （Apache License 2.0）的**二次开发版本**。在原版 0.1.2（commit `58f4b9e`）之上，
> 针对 **Qwen3.8-Flash-Next** 模型补充了双卡张量并行、1M 上下文扩展与图片理解能力。
> 上游项目版权归原作者所有，本仓库的修改同样以 Apache 2.0 发布。
>
> 上游项目简介：FreeToken 是一个边缘侧 MoE 大模型推理引擎，把 GPU、CPU、内存
> 作为统一的异构推理资源，让 100B+ 参数的模型能在消费级显卡上运行。

---

## 1. 相对上游的新增功能

| 功能 | 说明 |
|---|---|
| **双卡张量并行（TP=2）** | 为 `qwen4_exp` 架构（Qwen3.8-Flash-Next）打通 TP：MoE 专家按卡二分（每卡只加载/计算一半专家），注意力/GDN 投影按头分片，KV 头分片。专家权重的内存占用每卡减半（126 GB → 63 GB） |
| **1M 上下文扩展** | 新增 `--rope-scaling` 启动参数（YaRN），配合 `--max-seq-len-override` 把上下文从官方 256K 扩展到 1M tokens（off-label，实测见下文） |
| **图片理解** | 从零补齐多模态链路：ViT 视觉编码器（与 HF 参考实现逐位一致）、图片预处理、mrope 三维位置编码、embedding 注入、OpenAI 标准 `image_url` 请求格式。默认关闭，`FREETOKEN_LOAD_VISION=1` 开启 |
| **稳定性修复** | 超长图片请求改为 HTTP 400 优雅拒绝（原版会直接打崩调度进程）；修复 pip 版 NCCL 链接问题（TP 的前提）；offload 缓存槽位清零防 NaN |

---

## 2. 已验证环境与稳定性声明

**特别说明：目前只有 Qwen3.8-Flash-Next（NVFP4 量化版）完成了充分测试，
包括连续 12 小时长任务运行验证。** 其他模型（DeepSeek-V4-Flash、GLM、MiniMax 等）
未在本分支上验证过，请按原版能力对待。

| 环境 | 配置 | 状态 |
|---|---|---|
| 主测环境 A | 2 × NVIDIA RTX 5090 D（32 GB），247 GB 内存，CUDA 13.1，驱动 595.84 | 已完成单卡 256K、双卡 TP=2 + 1M + 图片理解测试，含 12 小时长任务 |
| 交叉验证环境 B | 2 × NVIDIA RTX 5880 Ada（48 GB），251 GB 内存，CUDA 13.0.2，驱动 580.95.05 | 已完成双卡 TP=2 + 1M + 图片理解测试，连续多请求稳定 |

> 环境 B 的 NCCL 拓扑与 A 不同（双卡跨 NUMA，SYS 互联），触发并修复了 TP=2 下的机内传输死锁问题，相关 workaround 已写进启动脚本，见第 5 节。

### 正确性验证

- TP=2 与单卡贪婪解码对比：4 条 prompt 中 3 条逐字一致，1 条仅末句浮点级差异
- ViT 视觉编码器与 HuggingFace 参考实现输出**逐位一致**（cosine = 1.0）
- mrope 位置编码与 HF `get_rope_index` 完全相等
- 长上下文"大海捞针"：248K tokens（官方上限内）与 **827K tokens**（1M 扩展）均命中
- 图片理解：正确识别测试图中的形状、颜色与文字；多轮带图对话正常
- 单元测试：新增 TP（10 项）+ 视觉（11 项）+ offload TP（7 项）；全量回归 1000+ 项通过

---

## 3. 环境要求

| 项目 | 单卡模式（256K） | 双卡模式（1M） |
|---|---|---|
| GPU | 1 × RTX 5090 / RTX 5880 Ada / 同级，≥ 24 GB 显存 | 2 张同型号或显存相近的 NVIDIA GPU |
| 内存 | ≥ 200 GB | ≥ 230 GB（模型 126 GB + PLE 表约 48 GB × 卡数 + KV 预留） |
| 磁盘 | ≥ 130 GB（模型 126 GB） | 同左 |
| 系统 | Linux x86_64，NVIDIA 驱动 r580+（CUDA 13），Python 3.10+ | 同左 |
| Python | 3.12（推荐） | 同左 |

> 模型文件：`~/models/Qwen3.8-Flash-Next-NVFP4`（126 GB，NVFP4 W4A4，仅路由专家量化，PLE 表 FP8，其余 BF16；量化方为 RadixArk）。

---

## 4. 安装

### 4.1 安装 CUDA 工具链（二选一）

FreeToken 依赖 torch 2.11 + cu130，因此需要 CUDA 13.x 的 `nvcc`。

**方案一：CUDA 13.0.2（不需要 rsqrt 补丁，交叉验证环境 B 使用）**

```bash
curl -LO https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
bash cuda_13.0.2_580.95.05_linux.run --silent --toolkit \
  --toolkitpath=$HOME/models/cuda-13.0 \
  --no-opengl-libs --no-man-page

export CUDA_HOME=$HOME/models/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
```

**方案二：CUDA 13.1（主测环境 A 使用，新版 glibc 需打 rsqrt 补丁）**

CUDA 13.1 的头文件与新版 glibc（如 Ubuntu 26.04）存在 `rsqrt` 声明冲突，
会导致 kernel JIT 编译失败。不碰系统文件的做法——复制一份打补丁：

```bash
mkdir -p ~/cuda-13.1-patched/targets/x86_64-linux
cd ~/cuda-13.1-patched
cp -r /usr/local/cuda-13.1/bin .
cp -r /usr/local/cuda-13.1/targets/x86_64-linux/include targets/x86_64-linux/
ln -s /usr/local/cuda-13.1/targets/x86_64-linux/lib targets/x86_64-linux/lib
sed -i 's|__device_builtin__ double                 rsqrt(double x);|__device_builtin__ double                 rsqrt(double x) __THROW;|; s|__device_builtin__ float                  rsqrtf(float x);|__device_builtin__ float                  rsqrtf(float x) __THROW;|' targets/x86_64-linux/include/crt/math_functions.h

export CUDA_HOME=~/cuda-13.1-patched
export PATH=$CUDA_HOME/bin:$PATH
```

（CUDA 13.2+ 已官方修复，可跳过补丁。）

### 4.2 安装 FreeToken

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"

# 拉取本仓库
git clone https://github.com/mr1528126360/FreeToken-qwen38.git ~/FreeToken
cd ~/FreeToken
git checkout feat/qwen4-exp-tp-vision

# 创建虚拟环境
uv venv --python 3.12 .venv

# 国内环境用 pip + 清华镜像（uv 会把 sglang-kernel 钉到 GitHub releases，国内可能不通）
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install -e ".[accel]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 海外环境可尝试：uv pip install -e ".[accel]"

# 验证
.venv/bin/ft --version   # freetoken version 0.1.2
```

> 注意：`.venv` 不要从其他机器直接拷贝，uv venv 依赖绝对路径。

### 4.3 带宽校准（每块 GPU 一次）

```bash
cd ~/FreeToken
.venv/bin/ft bench bw --gpu 0
.venv/bin/ft bench bw --gpu 1   # 双卡模式
```

结果保存在 `~/.cache/freetoken/benchbw/`。

### 4.4 下载模型

```bash
# Hugging Face
huggingface-cli download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4

# 国内可用 ModelScope
# modelscope download --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
#   --local_dir ~/models/Qwen3.8-Flash-Next-NVFP4
```

---

## 5. 运行与关闭

### 5.1 单卡模式（256K 上下文，官方上限内，质量最稳）

```bash
cd ~/FreeToken
export CUDA_HOME=$HOME/models/cuda-13.0   # 或 ~/cuda-13.1-patched
export PATH=$CUDA_HOME/bin:$PATH
export FREETOKEN_LOAD_VISION=1

nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --gpu 0 --port 1919 --host 0.0.0.0 \
  --num-tokens 262144 --moe-cache-size 1024 \
  > ~/freetoken-qwen-serve.log 2>&1 &
```

### 5.2 双卡模式（1M 上下文 + 图片理解）

在部分机器上（尤其是双卡跨 NUMA、无 P2P 稳定互联），NCCL 的 P2P/SHM 传输会死锁。
此时必须强制 NCCL 走 loopback socket：

```bash
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
```

> 环境 B（RTX 5880 Ada）必须加这三项；环境 A（RTX 5090 D）拓扑不同，未触发此问题，
> 但如果你的双卡也出现 TP=2 启动/请求随机挂死（GPU 100%、无进展），请优先尝试此 workaround。

完整启动命令：

```bash
cd ~/FreeToken
export CUDA_HOME=$HOME/models/cuda-13.0   # 或 ~/cuda-13.1-patched
export PATH=$CUDA_HOME/bin:$PATH
export FREETOKEN_LOAD_VISION=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket

nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 1919 --host 0.0.0.0 \
  --num-tokens 1048576 --max-seq-len-override 1048576 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
  --moe-cache-size 1024 --max-prefill-length 16384 \
  > ~/freetoken-qwen-serve.log 2>&1 &
```

关键参数说明：

- `--moe-cache-size 1024`：**必须显式指定**，否则 MoE 缓存自动吃满显存导致 KV 分配 OOM。TP 下该值是每卡的 slot 数。
- `--rope-scaling` + `--max-seq-len-override`：1M 为 YaRN×4 off-label 扩展，官方只背书 256K。
- `--max-prefill-length 16384`：带图请求必须单分块处理，此值覆盖任意尺寸图片。
- `--host 0.0.0.0`：监听局域网；去掉则仅本机。**服务无鉴权，勿暴露公网。**

### 5.3 使用启动脚本（推荐）

交叉验证环境 B 使用了一个独立启动脚本，便于维护：

```bash
# ~/models/start-qwen-nvfp4.sh
#!/bin/bash
export CUDA_HOME=$HOME/models/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export FREETOKEN_LOAD_VISION=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket

cd ~/models/FreeToken
nohup .venv/bin/ft serve \
  --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 1919 --host 0.0.0.0 \
  --num-tokens 1048576 --max-seq-len-override 1048576 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
  --moe-cache-size 1024 --max-prefill-length 16384 \
  > ~/models/freetoken-qwen-serve.log 2>&1 &
```

之后只需：

```bash
chmod +x ~/models/start-qwen-nvfp4.sh
~/models/start-qwen-nvfp4.sh
```

### 5.4 就绪判断

双卡加载约 5~8 分钟（126 GB 权重读盘 + 约 48 GB PLE 表 × 2 卡）：

```bash
.venv/bin/ft ctl health        # status=ok 即就绪
tail -f ~/freetoken-qwen-serve.log
```

### 5.5 关闭服务

```bash
pkill -f "venv/bin/ft serve"
sleep 10
nvidia-smi                       # 确认显存释放
```

若显存未释放（kill 后 `nvidia-smi` 仍有占用）：

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do kill -9 $p; done
```

---

## 6. 客户端配置

- 本机地址：`http://127.0.0.1:1919`
- 局域网地址示例：`http://10.38.6.53:1919`（以实际 IP 为准）
- OpenAI 兼容接口：`POST /v1/chat/completions`、`/v1/models`
- Anthropic 兼容接口：`/v1/messages`
- **无鉴权，勿暴露公网**

Trae / OpenAI 兼容客户端配置：

| 配置项 | 值 |
|---|---|
| Base URL | `http://<实际IP>:1919/v1` |
| API Key | 任意非空字符串，如 `sk-local` |
| 模型 ID | `Qwen3.8-Flash-Next-NVFP4` |
| Context Window | 1048576（双卡 1M）/ 262144（单卡 256K） |
| Max Output Tokens | 8192~32768（模型默认开思考，思考计入输出额度） |

curl 验证：

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"你好"}],"max_tokens":256}'
```

图片理解（需 `FREETOKEN_LOAD_VISION=1`）：

```json
{
  "model": "Qwen3.8-Flash-Next-NVFP4",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
        {"type": "text", "text": "描述这张图"}
      ]
    }
  ]
}
```

限制：仅支持图片（无 video/audio）；带图请求不进前缀缓存。

---

## 7. 实测性能

### 7.1 主测环境 A：2 × RTX 5090 D（CUDA 13.1）

| 指标 | 双卡 TP=2（1M 配置） | 单卡（256K 配置） |
|---|---|---|
| 输入 prefill | ~2240 tok/s | ~2100~3100 tok/s |
| 输出 decode | ~43.6 tok/s | ~41 tok/s |
| 显存 | 每卡 ~31 GB | ~23 GB |
| 内存 | 两进程合计 ~190 GB | ~125 GB |

### 7.2 交叉验证环境 B：2 × RTX 5880 Ada（CUDA 13.0.2，NCCL socket 传输）

| 指标 | 双卡 TP=2（1M 配置） | 单卡（262K 配置） |
|---|---|---|
| 输出 decode | ~19–23 tok/s | ~22–25 tok/s |
| TTFT（短 prompt） | 首请求 ~16 s（预热），之后 ~2 s | ~4–6 s |
| 显存 | 每卡 ~27 GB | ~23 GB |
| 内存 | 两进程合计 ~190 GB | ~125 GB |

5090 D 的显存带宽约为 5880 Ada 的 1.9 倍，decode 差距符合带宽比。
双卡的价值是**容量**（1M 上下文、专家/显存分片），速度与单卡基本持平——
与仓库文档在 5090 D 上的结论一致。

---

## 8. 常见问题与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| TP=2 启动卡死 / 请求随机挂死（GPU 100%、无进展） | NCCL 机内 P2P/SHM 传输在双 NUMA 等拓扑上死锁 | 启动前加 `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket`，或改用单卡模式 |
| `nvcc would build kernels linking libcudart.so.12` | `CUDA_HOME` 指向了 12.x | 确认 `CUDA_HOME=$HOME/models/cuda-13.0` 或 `~/cuda-13.1-patched` |
| 启动时报 `rsqrt` 冲突 | CUDA 13.1 + 新版 glibc | 使用打补丁的 `~/cuda-13.1-patched`，或改用 CUDA 13.0.2 |
| 显存 OOM | `--moe-cache-size` 未指定或过大 | 双卡 1M 推荐 `--moe-cache-size 1024`，单卡 256K 同样 |
| 和其他大模型服务同时运行时崩溃 | 内存/显存互相抢占 | 切换时先停掉另一个；Qwen 与 DeepSeek-V4-Flash 也不能同时运行 |
| 图片请求崩溃 / 无响应 | 图片过大或格式不支持 | 确保已加 `--max-prefill-length 16384`；超长图片会被优雅拒绝（HTTP 400） |

---

## 9. 注意事项

- **仅 Qwen3.8-Flash-Next 经过充分测试**（含 12 小时长任务）；其他模型未验证
- 1M 上下文是 YaRN off-label 扩展，官方只背书 256K；超长文本的生成质量请自行评估
- 图片理解仅支持图片（无 video/audio）；带图请求不进前缀缓存
- 不能与 DeepSeek-V4-Flash 等其他大模型服务同时运行（内存不足会 OOM 双杀）
- 详细设计/改动清单见 [改动文档-双卡TP与视觉.md](改动文档-双卡TP与视觉.md)
- 另一台机器的实测记录见 [RUN-Qwen3.8-NVFP4-FreeToken.md](RUN-Qwen3.8-NVFP4-FreeToken.md)

---

## 10. License

Apache License 2.0（与上游一致）。上游项目：[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)。
