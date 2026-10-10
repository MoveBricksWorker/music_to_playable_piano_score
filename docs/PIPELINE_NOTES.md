# 工程笔记

这份文档记录**为什么这么做**，以及支撑每个决定的实测数据。
面向要改这条管线的人；只想出谱的话看 [README](../README.md) 就够了。

---

## 环境：为什么不重新装，而是直接复制

用户问过「环境从原环境中直接复制粘贴不行吗」——可以，这就是整合包的做法
（[`bundle_env.py`](../bundle_env.py)）。比 pip 重装好在：不用下载数 GB、不需要网络、
与开发机完全一致。

| 项 | 复制后是否有效 | 原因 |
|---|---|---|
| `python.exe` | ✅ | Windows 上它是基础安装的副本；`sys.prefix` 由**可执行文件位置**推导，跟目录走 |
| `site-packages` | ✅ | 包按 `sys.prefix` 定位自己；`.pth` 里放的是相对路径 |
| torch 的 DLL | ✅ | `torch\lib` 由包自己按运行时路径加载 |
| **`Scripts\*.exe`** | ❌ | **写死了安装时的绝对路径**（`pip.exe` / `demucs.exe` 等） |
| `pyvenv.cfg` | ⚠️ | 指向 `home = <原基础解释器>`；把基础解释器一起复制即自洽 |

**唯一前提**：只用 `python.exe -m 模块名` / `python.exe 脚本.py`，不用 `Scripts\*.exe`。
本包所有代码都满足（`run_pipeline.py` 用 `sys.executable` + `-m`）。

实测（包内解释器，从新位置启动）：

```
executable: <包>\python\python.exe
prefix    : <包>\python
torch 2.6.0+cu124  cuda=True  NVIDIA GeForce RTX 4060 Laptop GPU
就绪 16 个包，缺失 0
```

复制后做了**保守瘦身**（`bundle_env.py --trim`），删掉管线不 import 的包，释放 **2.89 GB**：

```
tensorflow  1.29 GB   ~orch（torch 安装残留）1.12 GB   bitsandbytes 201 MB
cv2 108 MB   pyarrow 84 MB   pymupdf 51 MB   plotly 54 MB   + jupyter 系列
```

`matplotlib` / `seaborn` **没删**——它们在依赖链上，省几十 MB 不值得冒险。

---

## 模型搬运与离线

[`make_portable.py`](../make_portable.py) 把权重搬进 `models/`，搬运后**逐个校验
SHA-256 与大小**并写 `models/MANIFEST.json`（47 项全通过）。

| 模型 | 体积 | 用途 |
|---|---|---|
| `m-a-p/SheetSage2` | 218 MB | ④ 节拍 / 调号 / 和弦 / 旋律 |
| demucs `htdemucs` | 84 MB | ① 分离（4 轨，取 other） |
| demucs `htdemucs_6s` | 55 MB | ① 分离（6 轨，取 piano） |
| transkun `2.0.pt` | 56 MB | ③ 转谱（已随 wheel 在 site-packages 内） |

**没有搬** `m-a-p/MERT-v2-FullSong`（2.4 GB）：那是开发期实验用的音频编码器，
本管线**不 import 它**。

`run_pipeline.py` 的 `_apply_model_env()` 自动设置：

```
HF_HOME              = <包>/models/hf        （SheetSage2）
TORCH_HOME           = <包>/models/torch     （demucs）
HF_HUB_OFFLINE       = 1
TRANSFORMERS_OFFLINE = 1
```

只在包内确实存在对应目录时才改，否则保留系统默认。

### `HF_HUB_OFFLINE=1` 是必须的，不是可选优化

demucs 4.1.0 的加载顺序是：先向 HuggingFace 要 `htdemucs.yaml` + `htdemucs.safetensors`，
失败**才**回退到旧仓库用本地 `.th`。它没有「只用本地文件」的入口
（`repo=None` 不接受路径）。无网络时它仍会去连，实测：

| | 加载耗时（两个模型） |
|---|---|
| 不设 `HF_HUB_OFFLINE` | **284 秒**（全花在重试超时） |
| 设 `HF_HUB_OFFLINE=1` | **2.4 秒** |

---

## 各项判据背后的实测

### 起音位置漂移（`_split_gap` 为什么必须存在）

上游 `notation/score.py:149` 的 `_notate_span` 把跨度按"小节线 + **整拍**"切段再贪心填充。
在本项目的三连音编排上这会**移动起音**。实测（victim，12 tick/拍，2/4 拍）：

```
编排写的是 tick 0,2,4,6,8,10（三连音十六分，间距 2）
_notate_span(0, 2/12) 返回的第一个音时值是 5/6（= tick 10）
→ tick 10 被吃掉，tick 11 被凭空造出来
```

全曲统计：编排 1104 个起音里 **274 个丢失、285 个多出、226 个被移动 ±2 tick**。
±2 tick 正好是三连音十六分与直十六分的差，所以听感上就是"三连音变成切分"。

修法是 [`engrave._split_gap`](../pipeline/engrave.py#L967)：切点只取**小节线**，
因为 `end` 本身就是"同手下一个起音的起点"，只要时值不越过 `end`，起音位置一个都不会动。

### 三连音被写成切分（`dual_lattice` 的第二次修正）

按窗口"逐起音各自决定格子"会留下一个具体失效：一段三连音里只要有个别起音被划到 duple，
它就被吸到 3 tick 点上，前后间距从 2 变成 3——听到的正是"三连音被写成切分"。
实测记录（`run_best/diagnose_syncopation.py`）：victim 463 对三连音里 **132 对被写成切分
（28.5%）**，最常见形态 `played 2.0 -> written 3.0`。

修法：先找出**成串**的三连音格起音（原始间隔接近 2 或 4 tick 的连续段），整串统一按 triple 吸附。
判据只看原始间隔，不看四舍五入后的结果。

### 合并起音时不能用"小于一个格步"做判据

`merge_epsilon` 只合并**近似同时**的起音（转谱抖动 / 一上一下的装饰），单位是公共刻度。
不能用"小于一个格步"——三连音十六分的相邻格点正好相差一个格步（2/12 拍），
那样会把正常的连续音全并掉（实测会多并 705 个起音）。

而且判据必须对着**簇首**比，不能对着"上一个"：对着上一个会**链式吞并**，
只要每个相邻间隔都小于阈值，一整串连续起音会被并成一个
（实测 yoake 因此从 1889 起音并到 797，而真正"小于 0.2 tick"的只有 245 个）。

### 旋律 sidecar 只记旋律自己的音高

早先记的是"该 tick 上右手的整组音"，于是同刻伴奏音也被标成"不可动的保护音"，
可弹性修正无法执行——实测 victim 有 25 处跨度 20–27 的宽和弦折叠失败
（如 `[60, 85]`，其中只有 85 是旋律，60 却被一起保护了）。改对之后宽和弦从 94 降到 1。

### 读写必须共用同一套音名换算

`fix_playable` 早先自己写了 `pitch // 12 - 1`，对 MIDI 60 得到 `C-1`（应为 `C4`），
而读的时候用的是 `(octave + 1) * 12` —— 读写不对称，于是修正后的编排里出现
`C-12`、`E-12` 这种不存在的音，雕版后跨度反而变大（实测 victim 最大跨度从 40 变成 43）。
现在两个方向都复用 `ensemble2score.piano_arrange` 的同一对函数。

### 跨度折叠的终止条件曾经是错的

早先写成 `if lowered <= work[0] - 12: break`，那是错的（对某个音级 `pitch - 12`
永远不小于 `work[0] - 12`），保护从不触发，循环一路折到守卫上限、产出**负 MIDI 值**
（实测 `[60,64,69,81]` → `[-132,-128,81]`，写成音名是 `C-12 E-12`），雕版后跨度反而更大。

### 速度标记必须放进小节里

`notation/piano_score.py` 早先把 `MetronomeMark` 挂在 part 上。实测 music21 9.9.2
**完全不写进 MusicXML**（最小可复现用例里 `<sound tempo=>` 出现 0 次；放进小节里则为 1 次）。
后果：`<sound>`、`<metronome>`、`<direction>` 在成品里全部 0 个，任何记谱软件只能用默认
120 BPM 播放。victim 的真实速度是 **76.9 BPM**，于是导出的音频比 MIDI 快了
`120 / 76.9 ≈ 1.56` 倍——这正是"MP3 速度不对"。修法见
[`engrave.fix_tempo_marking`](../pipeline/engrave.py#L113)。

### MuseScore 4 的参数顺序

`notation/score.py` 原来的调用是 `[exe, "-o", target, musicxml]`，在 MuseScore 4.7 上会以
**1320** 退出（它不认这个参数顺序，只给出 "failed convert"）。加上 `-f pdf` 并把输入文件
放在 `-o` 之前，同一份谱子就能正常导出（实测 exit=0）。见
[`engrave.export_pdf_fixed`](../pipeline/engrave.py#L678)。

### 6 轨 piano + 4 轨 other 相加（为什么是这份音频）

`probe_piano_prominence.py` 实测 victim 上 6 轨的 `piano` 分轨在副歌掉 6 dB
（主歌 −20.4 dBFS、副歌 −26.7），而 4 轨的 `other` 在**同一处**涨 25.8 dB——那是
`htdemucs_6s` 把副歌伴奏路由到了自己的 other 头（Demucs 文档对 piano/guitar 这一对
正是这个警告）。`compare_by_section.py` 给出了分段后果：

```
副歌  6轨 piano  88 起音 / 88.2% 和弦音
      4轨 other 155 起音 / 76.9% 和弦音
      相加      147 起音 / 81.3% 和弦音
```

相加保留了 4 轨的音符数，和弦音占比更高，且在更安静段落没有可测损失。

### 为什么转谱不用 SheetSage2

SheetSage2 是 lead-sheet 模型，旋律头是**每个十六分音符一个类别分布**，
所以同一时刻最多出一个音。在本项目自己产出的所有分轨上实测：
每起音平均同时音数 = **1.00**（包括 piano）。复音钢琴谱必须用复音转谱器。

---

## 用电子乐试跑发现的一个静默失效（Laur – PRAGMATISM）

拿一首 artcore / 电子乐（120 s）试跑，暴露出 `fix_playable` 可以**在一条规则都没生效的
情况下报告成功**。

### 现象

```
谱面口径：宽和弦 74 处，最大跨度 64 半音      （判据是 ≤19）
fix_playable 的统计：R2 降八度 23 次、R2 因跨度删音 25 个 —— 看起来它干了活
```

### 根因：保护集覆盖整组时，折叠与删除同时失效

`fix_playable.fold_to_span` 的两道守卫都依赖"存在非旋律音"：

```python
for index, raise_octave in ((len(work) - 1, False), (0, True)):
    candidate = work[index]
    if candidate in keep: continue          # 旋律音不折
    ...
deletable = [i for i, p in enumerate(work) if p not in keep]
if not deletable:
    break                                    # 全是旋律音 → 放弃，且不报告
```

当某个 tick 的 `keep`（来自 `melody_protected`）等于该组全部音高时，
即使 `work[-1] - work[0] > HAND_SPAN` 也**什么都不做**：

```
fold_to_span([32, 44, 96], melody_top=96, reserved={32, 44, 96})  →  ([32,44,96], 0, 0)
fold_to_span([32, 44, 96], melody_top=96, reserved={96})          →  ([80,96],    4, 0)
```

### 实测规模

```
曲目          起音组   跨度>19   保护占满   其中超标的
victim         1396       0        0          0
yoake          1249       0        0          0
collapse       1012       0        0          0
blessing       1323       0        0          0
pragmatism      614      66       66         66
```

四首样例一个都没触发（SheetSage2 的真·人声旋律是单声部、音区集中）。
触发条件是**旋律来源稠密**。

### 对照实验：把旋律换成空 MIDI

```
宽和弦 0    拥挤起音 0    最大跨度 19    谱面 1070 音 / 856 起音    调号 -4    丢音 0
```

**这个曲子上的宽和弦不是转谱的锅，是"伪旋律 + 保护"这条链路的锅。**

### 为什么这次试跑的旋律来源是稠密的

包内 `transformers` 是 5.6.1，而 SheetSage2 的远程代码要求 4.45
（`BartDecoder(..., embed_tokens=...)` 在 5.x 已删除；精简版也没有 `.venv-ss2`），
所以标签是用 `work/_pragmatism_labels.py` 顶出来的，其中旋律取的是
**转谱每刻的最高音**。这有两个连锁后果：

1. `dedupe_piano` 会把这些音从钢琴轨删掉——912 个转谱音被删掉 **645 个（71%）**，
   因为"最高音"几乎是每个分解和弦最上面那个音；
2. 剩下的每个起音，其最高音都在 `melody_protected` 里，于是宽和弦全部被保护、修不动。

真·人声旋律在音高上**不重合于**钢琴轨最高音，所以正常路径不会这么糟。
这也反过来说明一条设计约束：**旋律必须来自独立声源**；
用钢琴轨自身提旋律，会让"去串音"那一步吃掉大部分素材。

### 顺带得到的正面结论

电子乐在这套管线上的**强项**恰恰是节拍与节奏族：

```
librosa 拍点间隔 0.511 s，σ = 0.0242（量化过的电子鼓）
双格量化判定 triple 格，49/49 个窗口一致（全程无需切换）
R1 = 0（没有落单三连音）
```

即：**鼓点是电子的，管线很稳；弱点在"谁是旋律"这个人声模型假设上**——
对没有人声的器乐曲，`melody_vocal` 这个前提本身就不成立。

---

## 三连音判据的密度偏置（已修）

### 症状

某首电子乐的转谱起音里只有 **51.3%** 落在三连音格点上（真三连音应为 92%，
纯随机为 50%），也就是**没有任何三连音证据**，但量化器判定整首为 `triple`，
501 个起音全部落在偶数 tick 上，谱面写满三连音。

### 根因：拿两套格的绝对代价直接比大小

```python
dup = sum(_error(p, DUPLE)  for p in pos[start:end])   # step=3 → 每拍 4 个格点
tri = sum(_error(p, TRIPLE) for p in pos[start:end])   # step=2 → 每拍 6 个格点
chosen = "triple" if tri + switch_penalty < dup else "duple"
```

对拍内均匀分布的起音，到最近格点的平均距离是 `step / 4`：

```
直十六分  3/4 = 0.75 tick
三连音    2/4 = 0.50 tick      ← 与音乐内容无关地白拿 0.25 tick
```

所以"`tri < dup`"这件事**在纯噪声上自动成立**。实测：

```
40 万个均匀随机起音
  直格平均代价 0.7507   三连格平均代价 0.5003   差 +0.2504
  逐个起音谁更近：三连格胜 50.0%，直格胜 16.6%

各曲实测差值（直格代价 − 三连格代价）
  victim       +0.2340   ← 贴着随机基准
  collapse     +0.0577
  blessing     +0.0317
  pragmatism   +0.2546   ← 与随机基准几乎相同
  纯随机        +0.2504   ← 基准
```

**victim 也贴着随机基准**——它选对三连音是碰巧（真三连音的曲子恰好有 92.2% 的起音
落在偶数 tick，所以答案对）。这就是这个缺陷长期没被发现的原因：唯一大量用三连音的
样例，正好也是真三连音。

### 修法

各自除以自己的随机基准（`step/4`）再比；迟滞从绝对 tick 数改成**相对的**每起音代价差。
默认 `cost_normalization="density"`、`switch_margin=0.01`；`"raw"` 保留旧行为。

```python
baseline = {DUPLE: DUPLE / 4, TRIPLE: TRIPLE / 4}
dup_c = dup / (len(chunk) * baseline[DUPLE])
tri_c = tri / (len(chunk) * baseline[TRIPLE])
chosen = "triple" if tri_c + switch_margin < dup_c else "duple"
```

### 效果（同一份输入，只换代价口径）

| 曲目 | 三连音音数 | R2 | R3 | R4 |
|---|---|---|---|---|
| victim | 997 → 989（−0.8%） | 90 → 89 | 128 → 127 | 254 → 256 |
| yoake | 685 → 649（−5.3%） | 95 → 88 | 265 → 254 | 340 → 315 |
| collapse | 506 → 444（−12.3%） | 70 → 58 | 159 → 148 | 230 → 210 |
| blessing | 354 → 255（−28.0%） | 35 → 22 | 113 → 88 | 200 → 160 |

量化器自报的"三连音窗口"数（判据越保守，这个数越小）：

```
victim      100/101 → 86/101    真三连音，基本不动
yoake        15/89  →  1/89
collapse     42/64  →  8/64
blessing     76/88  →  3/88
```

**真三连音的几乎不动、噪声曲子大幅回归、形状判据全线改善**——判据修对了应有的形状。

### 残留（未做听感验证）

* 最大吸附位移上升（yoake 1.26→1.45、blessing 1.16→1.49 tick）。旧口径为了贴合密格
  本来就把位移压低了，压的是错误方向；但"1.49 tick 是否比 1.26 更好听"要靠耳朵定。
* `switch_margin` 在 0.005–0.05 之间结果几乎一致（duple 占比变动 <1%），默认 0.01。
* 复现与对照：`work/_probe_bias.py`（随机基准）、`work/_probe_triplets3.py`
  （逐曲证据强度）、`work/_sweep_lattice.py` / `work/_sweep_scores.py`（口径对照）。

---

## 报告口径的坑（已修，但值得记住）

`run_pipeline.py` 的验收阶段曾经读 `bad_measures` / `duration_ratio` 这些
`report.json` 里**不存在**的字段。两个具体的坑：

1. `report.get("bad_measures", report.get("measures_off_length", "?"))`
   —— `.get` 的默认值参数会被**立即求值**，所以外层拿不到内层的结果，
   显示的 `?` 并不是"没数据"，而是"取错了 key"。
2. 雕版器去读 `deliverables/<曲>/run.json` 取时长，那个目录只在原始项目里存在，
   于是 `duration` 恒为 0、`notes_per_second` 恒为 0.0 —— 一个看起来有值、
   其实是常数的指标。

现在的做法：**只报真实存在的字段**，并在表头标明这是**编排层口径**，
谱面层的数据一律交给 `verify_playable` / `check_shapes` 去解析 MusicXML。

---

## 实测性能（RTX 4060 Laptop，离线）

出谱（四首，从 `--from transkun` 起）：

```
① 分离 33s（单首） → ② 相加 1s → ③ 转谱 30s（单首）
下游四首全跑（⑤⑥⑦⑧）约 0.6 分钟
```

`--from arrange --to engrave` 四首约 **0.2 分钟**（不需要 GPU）。
