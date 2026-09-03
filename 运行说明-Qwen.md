# FreeToken + Qwen3.8-Flash-Next-NVFP4 运行说明

本机部署：2 × RTX 5880 Ada Generation（48GB）双卡张量并行（TP=2，自研补丁）；
模型在 `/home/user/models/Qwen3.8-Flash-Next-NVFP4`（126GB，NVFP4 专家权重
offload 到内存、按双卡分片各 ~63GB；47.7GB PLE n-gram 表每卡各一份）。

> 本文 2026-09-02 已按本机实际环境校准：CUDA 用 `~/models/cuda-13.0`，
> JIT 编译 kernel 还需要 `ninja`（在 `/home/user/anaconda3/bin`，PATH 里必须有）。

> **重要：Qwen 和 DeepSeek-V4-Flash 不能同时运行**（内存不够，会被 OOM killer
> 双杀）。切换模型时先 `pkill -f "venv/bin/ft serve"` 再启动另一个。

## 启动（双卡 + 1M 上下文 + 图片理解）

```bash
export CUDA_HOME=~/models/cuda-13.0
export PATH=$CUDA_HOME/bin:/home/user/anaconda3/bin:$PATH   # ninja 必须能找到
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:
export FREETOKEN_LOAD_VISION=1        # 开启图片理解（不加则纯文本）
# TP 通信：这三个变量必须这样设，原因见下面"NCCL 传输"一节
export NCCL_P2P_DISABLE=1 NCCL_NET=Socket NCCL_SHM_DISABLE=1
cd ~/models/FreeToken

setsid nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 1919 --host 0.0.0.0 \
  --num-tokens 1048576 --max-seq-len-override 1048576 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
  --moe-cache-size 6144 --max-prefill-length 16384 > ~/models/freetoken-qwen-serve.log 2>&1 &
```

> 用 `setsid ... &` 而不是裸 `&`：服务是启动它的 shell 的子进程，
> 那个 shell 一旦被终止（例如脚本复用同一终端），裸后台进程会被一起带走。

参数说明：

- `--tp-size 2 --gpu 0,1`：双卡张量并行（本仓库 `feat/qwen4-exp-tp-vision` 分支的
  自研功能）。专家按卡分片（各载一半，内存/显存减半），KV 头分片。
- `--num-tokens 1048576`：KV 容量 1M tokens（双卡各 ~12.8GB）。
- `--max-seq-len-override` + `--rope-scaling`：把上下文从官方 256k 扩到 1M。
  注意：**官方只保证 256k**，1M 是 YaRN×4 的 off-label 扩展，超长文本的生成
  质量没有官方背书（needle 检索实测通过）。
- `--moe-cache-size 6144`：TP 下是**每卡** slot 数；不指定会被自动扩张吃满显存
  导致 KV 分配失败。每 slot ≈ 2.6 MiB，6144 slot = 15.9 GiB/卡 = 本机专家驻留
  25%（每卡 12288 个专家里驻 6144 个）。**这是解码速度最直接旋钮**：
  原值 1024（4.2%）时 95.8% 的专家每 token 都要走 PCIe 取，单流只有 22 tok/s。
  上限受 1M KV 挤占——`ft ctl cache` 显示重建预算 33.6 GiB，当前占用 29.9 GiB，
  再往上只剩 ~3.7 GiB（≈7400 slot），收益递减且会压缩 prefill 激活余量。
- 加载约 5~8 分钟（两卡并行读盘 + PLE 表）。

就绪判断：`ft ctl health` 显示 `status=ok`。

## 关闭

```bash
pkill -f "venv/bin/ft serve"
# 若显存未释放，按 PID 清理：
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  case "$(ps -o cmd= -p $p 2>/dev/null)" in /usr/*) ;; *) kill -9 $p;; esac
done
```

## 服务地址

- 本机 `http://127.0.0.1:1919`，局域网 `http://<服务器IP>:1919`
- OpenAI 兼容 `/v1/chat/completions`、`/v1/models`；Anthropic `/v1/messages`
- 无鉴权，勿暴露公网。

## 图片理解用法（需 FREETOKEN_LOAD_VISION=1）

```python
import base64, json, urllib.request
b64 = base64.b64encode(open("图片.png", "rb").read()).decode()
req = {
    "model": "Qwen3.8-Flash-Next-NVFP4",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "描述这张图"},
    ]}],
    "max_tokens": 300,
}
# POST 到 http://127.0.0.1:1919/v1/chat/completions
```

限制：仅图片（video/audio 不支持）；带图请求不进前缀缓存、不能跨 prefill
分块；端到端验证脚本在 `/tmp/vision_e2e.py`。

## 实测性能（TP=2 双卡，1M KV，2026-09-02）

流式测纯解码速率（`/home/user/ft_bench.py`，关思考模式，400 tokens/请求）：

| 并发 | `--moe-cache-size 1024`（旧） | `--moe-cache-size 6144`（现） |
|---|---|---|
| 单流 1 路 | 22.2 tok/s | **28.1 tok/s**（+27%） |
| 2 路聚合 | 31.1 tok/s | **48.4 tok/s** |
| 4 路聚合 | 44.5 tok/s | **85.3 tok/s**（+92%） |

长上下文回归：12 万 tokens 随机数字 haystack 埋针，检索正确（`/home/user/ft_longtest.py`，
端到端 ~960 tok/s，含客户端 tokenize + HTTP，引擎侧 prefill 仍在 2000 tok/s 量级）。
驻留率与解码速率近似线性：每少 100 次专家 miss ≈ 省 9.4 ms/token。

### NCCL 传输（重要，别再试了）

`ft ctl stats` 的 decode 只有 22~28 tok/s，但每 token 的耗时里 **~17 ms 花在 TP
all_reduce 上**：48 层 × 2 次 = 96 次 hidden(2560) all_reduce，socket 传输实测
176 µs/次。理论上更快的两种传输都不可用：

| 传输设置 | allreduce 延迟 | 结果 |
|---|---|---|
| `NCCL_P2P_DISABLE=1 NCCL_NET=Socket NCCL_SHM_DISABLE=1` | 176 µs | **唯一可用**（当前） |
| 去掉 `SHM_DISABLE`/`NET`（走 SHM） | 46 µs（3.8×快） | **卡在 CUDA graph 捕获**，11 分钟 0/3，服务起不来 |
| 默认（允许 P2P） | — | init 阶段直接挂死 |

两卡挂在不同 NUMA 节点（`nvidia-smi topo -m` 显示 `SYS`），跨 root complex 的
P2P 不可靠。想把这 17 ms 拿回来，得改 `kernel/pynccl.py` 的 graph capture 路径
（或换 NCCL 版本验证 SHM + cudagraph），不是调参数能解决的。

### 回退

改 `--moe-cache-size` 必须重启——**`ft ctl cache` 在线重建在 TP>1 下直接返回**
`HTTP 503: runtime rebuild unsupported under TP > 1`。启动前的原始命令记录在
`/home/user/ft_orig_cmd.txt`。

## 代码状态

以上 TP 和 vision 功能都在 `~/FreeToken` 的 `feat/qwen4-exp-tp-vision` 分支
（未 commit）。回退到官方原版：

```bash
cd ~/FreeToken && git stash && git checkout main
```

恢复本分支：`git checkout feat/qwen4-exp-tp-vision && git stash pop`。
