# FreeToken + Qwen3.8-Flash-Next-NVFP4 运行说明

本机部署：2 × RTX 5090 D（32GB）双卡张量并行（TP=2，自研补丁）；模型在
`/home/mr1528126360/models/Qwen3.8-Flash-Next-NVFP4`（126GB，NVFP4 专家权重
offload 到内存、按双卡分片各 ~63GB；47.7GB PLE n-gram 表每卡各一份）。

> **重要：Qwen 和 DeepSeek-V4-Flash 不能同时运行**（内存不够，会被 OOM killer
> 双杀）。切换模型时先 `pkill -f "venv/bin/ft serve"` 再启动另一个。

## 启动（双卡 + 1M 上下文 + 图片理解）

```bash
export CUDA_HOME=~/cuda-13.1-patched
export PATH=$CUDA_HOME/bin:$PATH
export FREETOKEN_LOAD_VISION=1        # 开启图片理解（不加则纯文本）
cd ~/FreeToken

nohup .venv/bin/ft serve --model ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 2 --gpu 0,1 --port 1919 --host 0.0.0.0 \
  --num-tokens 1048576 --max-seq-len-override 1048576 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
  --moe-cache-size 1024 --max-prefill-length 16384 > ~/freetoken-qwen-serve.log 2>&1 &
```

参数说明：

- `--tp-size 2 --gpu 0,1`：双卡张量并行（本仓库 `feat/qwen4-exp-tp-vision` 分支的
  自研功能）。专家按卡分片（各载一半，内存/显存减半），KV 头分片。
- `--num-tokens 1048576`：KV 容量 1M tokens（双卡各 ~12.8GB）。
- `--max-seq-len-override` + `--rope-scaling`：把上下文从官方 256k 扩到 1M。
  注意：**官方只保证 256k**，1M 是 YaRN×4 的 off-label 扩展，超长文本的生成
  质量没有官方背书（needle 检索实测通过）。
- `--moe-cache-size 1024`：TP 下是**每卡** slot 数；不指定会被自动扩张吃满显存
  导致 KV 分配失败。
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

- 本机 `http://127.0.0.1:1919`，局域网 `http://10.38.6.56:1919`
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

## 实测性能（TP=2 双卡）

| 指标 | TP=2 双卡 | TP=1 单卡 |
|---|---|---|
| 输入 prefill | ~1900~2250 tok/s | ~2100~3100 tok/s |
| 输出 decode | ~44 tok/s | ~41 tok/s |
| VRAM | 每卡 ~26GB（1M KV 配置） | ~23GB（256k 配置） |
| 内存 | 两进程合计 ~186GB | ~125GB |

TP=2 的主要收益是**显存/内存容量**（1M 上下文可行、专家缓存分片），
速度基本持平（all_reduce 开销抵消了带宽收益）。

## 代码状态

以上 TP 和 vision 功能都在 `~/FreeToken` 的 `feat/qwen4-exp-tp-vision` 分支
（未 commit）。回退到官方原版：

```bash
cd ~/FreeToken && git stash && git checkout main
```

恢复本分支：`git checkout feat/qwen4-exp-tp-vision && git stash pop`。
