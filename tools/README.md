# 诊断探针

这些脚本**不是管线的一部分**，跑管线不需要它们。它们是一次性诊断工具，
留着是因为它们能复现 README「已知缺陷」里的每一个数字——
以后改管线时，用它们确认自己没有把已经修好的东西改回去。

四个都是**只读**的：只解析产物，不写任何管线文件
（`trace_key_loss.py` 会在 `work/` 下临时写一份小谱面，跑完自己删）。

| 脚本 | 查什么 | 什么时候用 |
|---|---|---|
| [`check_structure.py`](check_structure.py) | 调号在不在、小节是否填满、编排里的起音有没有在谱面上丢掉、连音线碎片有多少 | 出谱之后的第一道体检；"谱面看起来不对"时先跑它 |
| [`diff_arrangement_vs_score.py`](diff_arrangement_vs_score.py) | 逐小节逐起音对齐编排与谱面，给出**具体例子** | `check_structure` 说"丢了 N 个起音"之后，用它定位 |
| [`check_accidentals.py`](check_accidentals.py) | 写出的临时记号里有多少是调号已经覆盖的（多余的） | 改 `_pitch_with_spelling` 前后对比 |
| [`trace_key_loss.py`](trace_key_loss.py) | `key.lab` 解析出的调名、五度数、拼写表；以及调号能否被 music21 导出 | 怀疑调号/拼写有问题时 |

用法：

```bash
python tools/check_structure.py
python tools/check_structure.py --root work/_review_fixed
python tools/diff_arrangement_vs_score.py --song victim --show 10
python tools/check_accidentals.py
python tools/trace_key_loss.py
```

它们和管线的两个验收器分工不同：

| | 回答的问题 |
|---|---|
| `pipeline/verify_playable.py` | **能不能弹**（跨度、同时音数） |
| `pipeline/check_shapes.py` | **好不好读**（三连音成组、时值形状 R1–R4） |
| `tools/check_structure.py` | **内容对不对**（调号、小节完整性、有没有丢音） |
