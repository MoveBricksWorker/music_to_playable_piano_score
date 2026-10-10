# 第三方组件与模型许可

本仓库自己的代码以 [MIT](LICENSE) 发布。**但它不包含、也不覆盖**它调用的第三方组件。

这份清单的依据是代码里的登记表 [`ensemble2score/io/registry.py`](ensemble2score/io/registry.py)——
那张表是**唯一权威来源**，改代码时请同步改这里。

---

## 一、随本仓库分发的第三方依赖：**无**

仓库里只有本项目的源码、文档，以及四首样例的**谱面产物**（`out/`，MusicXML / MIDI / PDF）。
不含任何模型权重、不含任何解释器或第三方库的代码。

Python 依赖通过 `pyproject.toml` / `install_env.py` 由使用者自己安装，
各归各的许可证。

---

## 二、管线实际调用的组件

| 组件 | 阶段 | 许可证 | 类别 | 备注 |
|---|---|---|---|---|
| [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | ④ 节拍/调号/和弦/旋律 | **CC-BY-NC-4.0** | ⚠️ **非商业** | 权重与代码均为非商业；其父模型 MERT-v2-FullSong 同样是 CC-BY-NC-4.0 |
| [Demucs](https://github.com/facebookresearch/demucs) | ① 音源分离 | MIT | 宽松 | `htdemucs` / `htdemucs_6s` 权重由官方渠道分发 |
| [Transkun](https://github.com/Yujia-Yan/Transkun) | ③ 复音钢琴转谱 | MIT | 宽松 | `2.0.pt` 随 PyPI wheel 分发 |
| [music21](https://web.mit.edu/music21/) | ⑦ 记谱 | BSD-3-Clause | 宽松 | |
| [pretty_midi](https://github.com/craffel/pretty-midi) | ⑤ MIDI 读写 | MIT | 宽松 | |
| [librosa](https://librosa.org) | 音频工具 | ISC | 宽松 | |
| [soundfile](https://python-soundfile.readthedocs.io) | 音频读写 | BSD-3-Clause | 宽松 | |
| [NumPy](https://numpy.org) / [SciPy](https://scipy.org) | 数值 | BSD-3-Clause | 宽松 | |
| [PyTorch](https://pytorch.org) | ①③ 推理 | BSD-3-Clause | 宽松 | |
| [MuseScore 4](https://musescore.org) | ⑧ 渲染 PDF | **GPL-3.0** | copyleft | **只作为外部程序调用**，不链接、不分发；不装它也能出 MusicXML/MIDI |
| [transformers](https://github.com/huggingface/transformers) | ④ 重新推理时 | Apache-2.0 | 宽松 | 只有 `--force` 重跑 SheetSage2 才需要 |

---

## 三、⚠️ SheetSage2 的非商业条款，实际意味着什么

**能做的**

- 用这条管线为自己学习、研究、做谱面参考；
- 把谱面产物（`out/` 里那些）用于非商业用途；
- 把整合包（含 SheetSage2 权重）在非商业前提下转发给别人。

**不能直接做的**

- 拿跑出来的谱面去做商业发行、商业编曲服务、付费产品；
- 把 SheetSage2 权重用于商业服务。

**要商用怎么办**：把阶段 ④ 换掉。管线对节拍/调号来源的依赖是**松耦合**的——
`work/sheetsage2/<曲>/` 下只要满足这几个文件就能跑：

```
beat.lab          每行一个拍点时间（秒）
downbeat.lab      每行一个强拍时间（秒）
key.lab           每行 "起始秒 <TAB> 结束秒 <TAB> 调名"（如 0.01  149.74  F:minor）
melody_vocal.mid  人声旋律（单音 MIDI）
```

替换实现见 `ensemble2score/io/registry.py` 里登记的其他候选
（`beat-this` 做拍点、`basic-pitch` / `pyin` 做旋律等，各自许可证一并登记在那里）。

---

## 四、关于四首样例曲目

`out/` 里是算法对以下四首既有乐曲的**转谱结果**，用于演示与回归验证：

| 目录 | 曲目 |
|---|---|
| `victim` | Mili – Victim |
| `yoake` | M2U, DAZBEE – 夜明けの歌 |
| `collapse` | Laur, 藍月なくる – Collapse |
| `blessing` | FictionJunction – blessing |

**音频文件不在本仓库内**，谱面是算法输出（不是原曲的官方谱）。
版权属于各自的权利人；这里仅作技术演示。如果你是权利人并希望移除，请开 issue。

---

## 五、如果你要重新分发整合包

整合包里有 SheetSage2 的权重（`models/hf/`），因此分发时请：

- 保留 [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) 的署名与非商业条件；
- 明确说明权重来自 <https://huggingface.co/m-a-p/SheetSage2>；
- 说明其中的 Demucs / Transkun 权重各自的来源与许可（见上表）。
