# Qwen3.8-Flash-Next-NVFP4 本地运行文档（FreeToken 引擎）

本文档记录在本机（2 × RTX 5880 Ada 48GB）用 FreeToken 部署
`~/models/Qwen3.8-Flash-Next-NVFP4`（126GB，NVFP4 量化，180B MoE）的流程与实测。

> 模型量化方为 RadixArk（NVFP4 W4A4，仅路由专家量化，PLE 表 FP8，其余 BF16），
> 官方评测 GSM8K 97.27% / AIME26 98.75%，与 BF16 参考持平。
> 推理引擎为 `~/models/FreeToken`（分支 `feat/qwen4-exp-tp-vision`），
> 特点是专家权重 offload 到内存、按需取到 GPU，消费级显卡即可跑 100B+ 模型。

## 1. 与 FreeToken 文档环境的差异

| 项目 | 仓库文档环境 | 本机 |
|---|---|---|
| GPU | 2 × RTX 5090 D 32GB（Blackwell） | 2 × RTX 5880 Ada 48GB（Ada，sm_89） |
| 内存 | 247GB | 251GB |
| CUDA | 13.1（手工打 rsqrt 补丁） | **13.0.2（安装在 `~/models/cuda-13.0`，无需补丁）** |

**双卡 TP=2 已可用**（需 NCCL 传输修复，见下）。最终配置：双卡 TP=2 + 1M 上下文
（YaRN×4 off-label）+ 图片理解。

### TP=2 挂死问题：根因与修复（2026-09-01 排查记录）

- **现象**：TP=2 启动时在 CUDA graph 捕获阶段两个 rank 双双挂死（GPU 100%、CPU 自旋、
  无任何进展）；禁用 graph 能启动且第一个请求正常，但后续请求仍随机挂死。
- **定位过程**：NCCL 调试日志显示两个 rank 都发出了完全一致的第一个
  `ncclAllReduce`（opCount 0, count 10240），但永不完成；`PYTHONFAULTHANDLER=1` +
  SIGABRT 抓栈显示主线程并不在 NCCL 调用里，而是堵在 PLE 哈希代码（ple.py:452）——
  实际是永不完成的 allreduce kernel 占着 GPU 流，后续所有 kernel 排队，主机线程
  堵在 CUDA 入队路径上自旋。
- **根因**：本机双卡分别挂在**两个 NUMA 节点**（`nvidia-smi topo -m` 显示 SYS 互联，
  跨 UPI），NCCL 的机内快速传输（GPU 直连 P2P 与主机共享内存 SHM）在这个拓扑上
  会合（rendezvous）不稳定/死锁。单独 `NCCL_P2P_DISABLE=1` 不够（回退到 SHM 仍挂）。
- **修复**：强制 NCCL 走 loopback socket 传输，启动前加三个环境变量（已写进启动脚本）：

  ```bash
  export NCCL_P2P_DISABLE=1
  export NCCL_SHM_DISABLE=1
  export NCCL_NET=Socket
  ```

- **效果**：TP=2 + CUDA graph 正常启动（图捕获约 1 分钟），连续多请求、700-token
  prefill 全部通过；decode ~19-23 tok/s，与单卡基本持平（allreduce 走 socket 的
  开销被双卡分片收益抵消）。仓库文档环境（5090 D）拓扑不同，未触发此问题。

## 2. 安装（已完成的步骤）

```bash
# uv + Python 3.12 venv（注意：不要用从其他机器拷贝的 .venv，uv venv 依赖绝对路径）
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
cd ~/models/FreeToken
uv venv --python 3.12 .venv

# CUDA 13.0 toolkit（torch 2.11+cu130 要求 nvcc 同为 13.x，装在项目目录内）
curl -LO https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
bash cuda_13.0.2_linux.run --silent --toolkit --toolkitpath=$HOME/models/cuda-13.0 \
  --no-opengl-libs --no-man-page

# 安装引擎（国内用清华镜像；.venv 里需先 python -m ensurepip）
export CUDA_HOME=$HOME/models/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install -e ".[accel]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 带宽校准（每块 GPU 一次，结果存 ~/.cache/freetoken/benchbw/）
.venv/bin/ft bench bw --gpu 0
.venv/bin/ft bench bw --gpu 1
```

## 3. 启动与停止

**启动（就这一条命令）：**

```bash
~/models/start-qwen-nvfp4.sh
```

**停止：**

```bash
pkill -f "venv/bin/ft serve"
```

若显存未释放（kill 后 nvidia-smi 仍有占用）：

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do kill -9 $p; done
```

**就绪判断**（加载约 7 分钟：126GB 权重读盘 + 47.7GB PLE 表 × 2 卡）：

```bash
cd ~/models/FreeToken && .venv/bin/ft ctl health    # status=ok 即就绪
tail -f ~/models/freetoken-qwen-serve.log           # 看日志
```

脚本内容（`~/models/start-qwen-nvfp4.sh`）：双卡 TP=2、端口 1919、监听 0.0.0.0、
1M 上下文（YaRN×4 off-label，官方只背书 262K）、`--moe-cache-size 1024`（每卡 slot 数，
必须显式指定）、`FREETOKEN_LOAD_VISION=1`（图片理解）、NCCL socket 传输三件套
（根因见第 1 节）。

**附录：单卡备用模式**（官方 262K 上下文，VRAM 占用更少；不需要 NCCL 修复）：

```bash
export CUDA_HOME=$HOME/models/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH FREETOKEN_LOAD_VISION=1
cd ~/models/FreeToken
nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --gpu 0 --port 1919 --host 0.0.0.0 \
  --num-tokens 262144 --moe-cache-size 1024 > ~/models/freetoken-qwen-serve.log 2>&1 &
```

## 4. 服务地址与客户端配置

- 本机 `http://127.0.0.1:1919`，局域网 **`http://10.38.6.53:1919`**
- OpenAI 兼容 `POST /v1/chat/completions`、`/v1/models`；Anthropic `/v1/messages`
- **无鉴权，勿暴露公网**

Trae / OpenAI 兼容客户端配置三项：

| 配置项 | 值 |
|---|---|
| API 地址 / Base URL | `http://10.38.6.53:1919/v1` |
| API Key | 任意非空字符串，如 `sk-local` |
| 模型 ID | `Qwen3.8-Flash-Next-NVFP4` |

curl 验证：

```bash
curl http://10.38.6.53:1919/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"你好"}],"max_tokens":256}'
```

图片理解（需 `FREETOKEN_LOAD_VISION=1`）：OpenAI 多模态格式，`content` 数组中放
`{"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}}`。
限制：仅图片（无 video/audio）；带图请求不进前缀缓存。

## 5. 实测性能（RTX 5880 Ada，非思考模式）

| 指标 | 双卡 TP=2（1M 配置，socket 传输） | 单卡（262K 配置） | 文档参考（5090 D 双卡） |
|---|---|---|---|
| decode | ~19–23 tok/s | ~22–25 tok/s | ~44 tok/s |
| TTFT（短 prompt） | 首请求 ~16s（预热），之后 ~2s | ~4–6 s | — |
| 显存 | 每卡 ~27 GB | GPU0 ~23 GB | 每卡 ~26 GB |
| 内存 | 两进程合计 ~190 GB | ~125 GB | ~186 GB |
| 上下文 | 1048576（YaRN×4 off-label） | 262144（官方上限） | 同左 |

5090 D 显存带宽约为 5880 Ada 的 1.9 倍，decode 差距符合带宽比。
双卡的价值是**容量**（1M 上下文、专家/显存分片），速度与单卡基本持平——
与仓库文档在 5090 D 上的结论一致。
模型默认开思考（thinking），客户端 max output tokens 建议 8192 以上。

## 6. 常见问题

- **TP=2 启动卡死/请求随机挂死（GPU 100% 无进展）**：NCCL 机内传输死锁，
  必须用第 1 节的 socket 传输三件套（`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
  NCCL_NET=Socket`）；或改用单卡模式（无此问题）。
- **`nvcc would build kernels linking libcudart.so.12`**：CUDA_HOME 指向了 12.x，
  必须 export `CUDA_HOME=$HOME/models/cuda-13.0`（torch 2.11 是 cu130）。
- **不能和其他大模型服务（如 llama-server GGUF 版）同时运行**：两者都要占大量
  内存/显存，会互相 OOM。切换时先停掉另一个。
- **Qwen 与 DeepSeek-V4-Flash 也不能同时运行**（FreeToken 文档原注）。
