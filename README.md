# FreeToken-Qwen3.8：双卡 TP / 1M 上下文 / 图片理解增强版

> 本项目是基于 [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
> （Apache License 2.0）的**二次开发版本**，在原版 0.1.2（commit `58f4b9e`）之上
> 针对 **Qwen3.8-Flash-Next** 模型做了三项增强。上游项目版权归原作者所有，
> 本仓库的修改同样以 Apache 2.0 发布。
>
> 上游项目简介：FreeToken 是一个边缘侧 MoE 大模型推理引擎，把 GPU、CPU、内存
> 作为统一的异构推理资源，让 100B+ 参数的模型能在消费级显卡上运行。

## 相对上游的新增功能

| 功能 | 说明 |
|---|---|
| **双卡张量并行（TP=2）** | 为 qwen4_exp 架构（Qwen3.8-Flash-Next）打通 TP：MoE 专家按卡二分（每卡只加载/计算一半专家），注意力/GDN 投影按头分片，KV 头分片。专家权重内存占用每卡减半（126GB→63GB） |
| **1M 上下文扩展** | 新增 `--rope-scaling` 启动参数（YaRN），配合 `--max-seq-len-override` 把上下文从官方 256k 扩展到 1M tokens（off-label，实测见下文） |
| **图片理解** | 从零补齐多模态链路：ViT 视觉编码器（与 HF 参考实现逐位一致）、图片预处理、mrope 三维位置编码、embedding 注入、OpenAI 标准 `image_url` 请求格式。默认关闭，`FREETOKEN_LOAD_VISION=1` 开启 |
| 稳定性修复 | 超长图片请求改为 HTTP 400 优雅拒绝（原版会直接打崩调度进程）；修复 pip 版 NCCL 链接问题（TP 的前提）；offload 缓存槽位清零防 NaN |

## 测试情况

**特别说明：目前只有 Qwen3.8-Flash-Next（NVFP4 版）完成了充分测试，
包括连续 12 小时长任务运行验证。** 其他模型（DeepSeek-V4、GLM、MiniMax 等）
未在本分支上验证过，请按原版能力对待。

测试环境：2 × NVIDIA RTX 5090 D（32GB），247GB 内存，CUDA 13.1，驱动 595.84。

### 正确性

- TP=2 与单卡贪婪解码对比：4 条 prompt 中 3 条逐字一致，1 条仅末句浮点级差异
- ViT 视觉编码器与 HuggingFace 参考实现输出**逐位一致**（cosine = 1.0）
- mrope 位置编码与 HF `get_rope_index` 完全相等
- 长上下文"大海捞针"：248k tokens（官方上限内）与 **827k tokens**（1M 扩展）均命中
- 图片理解：正确识别测试图中的形状、颜色与文字；多轮带图对话正常
- 单元测试：新增 TP（10 项）+ 视觉（11 项）+ offload TP（7 项）；
  全量回归 1000+ 项通过

### 性能（双卡 TP=2 vs 单卡）

| 指标 | 双卡 TP=2（1M 配置） | 单卡（256k 配置） |
|---|---|---|
| 输入 prefill | ~2240 tok/s | ~2100~3100 tok/s |
| 输出 decode | ~43.6 tok/s | ~41 tok/s |
| 显存 | 每卡 ~31GB | ~23GB |
| 内存 | 两进程合计 ~190GB | ~125GB |

双卡的价值是**容量**（1M 上下文、内存分片），速度基本持平。

## 环境要求

| 项目 | 单卡模式（256k） | 双卡模式（1M） |
|---|---|---|
| GPU | 1 × RTX 5090（32GB）或同级 | 2 张，建议同型号 |
| 内存 | ≥ 200GB | ≥ 230GB |
| 磁盘 | ≥ 130GB（模型 126GB） | 同左 |
| 系统 | Linux x86_64，NVIDIA 驱动 r580+（CUDA 13），Python 3.10+ | 同左 |

## 安装

### 1. CUDA 工具链补丁（CUDA 13.1 + 新版 glibc 必做）

CUDA 13.1 的头文件与新版 glibc（如 Ubuntu 26.04）存在 `rsqrt` 声明冲突，
会导致 kernel JIT 编译失败。不碰系统文件的做法——复制一份打补丁：

```bash
mkdir -p ~/cuda-13.1-patched/targets/x86_64-linux
cd ~/cuda-13.1-patched
cp -r /usr/local/cuda-13.1/bin .
cp -r /usr/local/cuda-13.1/targets/x86_64-linux/include targets/x86_64-linux/
ln -s /usr/local/cuda-13.1/targets/x86_64-linux/lib targets/x86_64-linux/lib
sed -i 's|__device_builtin__ double                 rsqrt(double x);|__device_builtin__ double                 rsqrt(double x) __THROW;|; s|__device_builtin__ float                  rsqrtf(float x);|__device_builtin__ float                  rsqrtf(float x) __THROW;|' targets/x86_64-linux/include/crt/math_functions.h
```

（CUDA 13.2+ 已官方修复，可跳过。）后续所有命令都需要：

```bash
export CUDA_HOME=~/cuda-13.1-patched
export PATH=$CUDA_HOME/bin:$PATH
```

### 2. 安装 FreeToken

```bash
# uv（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"

git clone <本仓库地址> && cd FreeToken
git checkout feat/qwen4-exp-tp-vision

uv venv --python 3.12 && source .venv/bin/activate

# 国内环境用 pip + 镜像（uv 会把 sglang-kernel 钉到 GitHub releases，国内不通）
python -m ensurepip
python -m pip install -e ".[accel]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 海外环境可直接：uv pip install -e ".[accel]"

ft --version   # 验证：freetoken version 0.1.2
```

### 3. 带宽校准（每块 GPU 一次）

```bash
ft bench bw --gpu 0
ft bench bw --gpu 1   # 双卡模式
```

### 4. 下载模型

```bash
huggingface-cli download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4
# 国内可用 ModelScope：
# modelscope download --model RadixArk/Qwen3.8-Flash-Next-NVFP4 \
#   --local_dir ~/models/Qwen3.8-Flash-Next-NVFP4
```

## 运行

### 单卡模式（256k 上下文，官方上限内，质量最稳）

```bash
nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --gpu 0 --port 1919 --host 0.0.0.0 \
  --num-tokens 262144 --moe-cache-size 1024 > ~/freetoken-qwen-serve.log 2>&1 &
```

### 双卡模式（1M 上下文 + 图片理解）

```bash
export FREETOKEN_LOAD_VISION=1
nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 1919 --host 0.0.0.0 \
  --num-tokens 1048576 --max-seq-len-override 1048576 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
  --moe-cache-size 1024 --max-prefill-length 16384 > ~/freetoken-qwen-serve.log 2>&1 &
```

关键参数：

- `--moe-cache-size 1024`：**必须显式指定**，否则 MoE 缓存自动吃满显存导致
  KV 分配 OOM。TP 下该值是每卡的 slot 数
- `--rope-scaling` + `--max-seq-len-override`：1M 为 YaRN×4 off-label 扩展
- `--max-prefill-length 16384`：带图请求必须单分块处理，此值覆盖任意尺寸图片
- `--host 0.0.0.0`：监听局域网；去掉则仅本机。**服务无鉴权，勿暴露公网**

加载约 5~8 分钟，`.venv/bin/ft ctl health` 显示 `status=ok` 即就绪。

### 关闭

```bash
pkill -f "venv/bin/ft serve"
sleep 10
nvidia-smi   # 确认显存释放；有残留 worker 则按 PID kill -9
```

## 使用

- OpenAI 兼容接口：`POST /v1/chat/completions`、`/v1/models`；
  Anthropic：`/v1/messages`
- 图片请求（OpenAI 多模态格式，需 `FREETOKEN_LOAD_VISION=1`）：

```json
{"model": "Qwen3.8-Flash-Next-NVFP4", "messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
  {"type": "text", "text": "描述这张图"}
]}]}
```

- 客户端建议：context window 填 1048576（双卡）/ 262144（单卡）；
  max output tokens 填 8192~32768（模型默认开思考，思考计入输出额度）

## 注意事项

- **仅 Qwen3.8-Flash-Next 经过充分测试**（含 12 小时长任务）；其他模型未验证
- 1M 上下文是 YaRN off-label 扩展，官方只背书 256k；超长文本的生成质量请自行评估
- 图片理解仅支持图片（无 video/audio）；带图请求不进前缀缓存
- 不能与 DeepSeek-V4-Flash 等其他大模型服务同时运行（内存不足会 OOM 双杀）
- 详细设计/改动清单见 [改动文档-双卡TP与视觉.md](改动文档-双卡TP与视觉.md)，
  运维手册见 [部署文档.md](部署文档.md) 和 [运行说明-Qwen.md](运行说明-Qwen.md)

## License

Apache License 2.0（与上游一致）。上游项目：
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)。
