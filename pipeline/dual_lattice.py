"""双格量化器：把转谱的起音吸附到 duple / triple **两套**格子上，而不是一张固定格子。

为什么需要它（本轮的诊断结论）
------------------------------------------------
项目原先的钢琴路径把 tick 网格写死成 4（后提高到 12），然后**每个起音独立四舍五入**到
那张格子。实测 victim 各轨相加的转谱成果：

    拍内位置分布      0.146/0.161/0.173 …（每拍约 6 个起音，间隔 ≈1/6 拍）
    最常见的 tick 余数 (0,474) (6,399) (12,214) (4,146) (2,130) (10,129) (8,121)

也就是说演奏的是**三连音十六分**（每拍 6 个），而直十六分是每拍 4 个。这两套格子：

    直十六分点  0/12, 3/12, 6/12, 9/12
    三连音点    0/12, 2/12, 4/12, 6/12, 8/12, 10/12

在同一条 12 tick 线上**互相冲突**：2/12 是三连音十六分、3/12 是直十六分。逐起音独立
四舍五入会把 0.153、0.168、0.192 三个起音一起压到 2/12 —— **三个音变一个音，三连音
就"没有了"**。把分辨率从 4 提到 12 只是换了个地方丢，没有解决冲突。

正确做法是**按局部上下文在两套格子里选一套**，再吸附到那一套的格点上：

  1. 以滑动窗口看局部起音落在哪套格子上代价更低；
  2. 窗口投票 + 迟滞，避免一个歧义间隔就翻转；
  3. 选定的格子上，间隔小于一步的起音**合并**（保留音高并集），其余**绝对吸附**。

绝对吸附（而不是相对上一个已吸附位置）是关键：相对吸附会把每个音的误差累积下去，
`work/_triplet_sprint/RESULTS.md` 记录过中段累积到 105 tick 的漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 一拍内的候选格位数。duple = 4 个直十六分；triple = 6 个三连音十六分。
#: 两者取最小公倍数 12 作为公共刻度，所以每个格点都能用 1/12 拍精确表示。
COMMON = 12
DUPLE = 3          # 12/4：直十六分每 3 个公共刻度
TRIPLE = 2         # 12/6：三连音十六分每 2 个公共刻度


@dataclass
class Quantized:
    """量化结果。每个字段都能被独立复核，不是黑盒。"""

    onsets: list[tuple[int, list[int]]]     # (公共刻度, 音高列表)，已排序
    lattice: str                            # "duple" / "triple" / "mixed"
    duple_share: float
    merged: int                             # 被合并掉的起音数
    max_shift_ticks: float                  # 最大位移（公共刻度）
    per_window: list[str] = field(default_factory=list)


def _nearest_on_lattice(position: float, step: int) -> int:
    """把拍内位置（单位：公共刻度，12 = 一拍）吸到 ``step`` 格位上。"""
    return int(round(position / step)) * step


def _error(position: float, step: int) -> float:
    """吸附到 ``step`` 格位的代价：格点越近代价越小，且不奖励"合并"。"""
    return abs(position - _nearest_on_lattice(position, step))


def quantize(
    positions: list[float],
    pitches: list[list[int]],
    *,
    window: int = 24,
    hop: int = 12,
    switch_penalty: float = 0.02,
    merge_epsilon: float = 0.2,
    cost_normalization: str = "density",
    switch_margin: float = 0.01,
    snap_policy: str = "family",
) -> Quantized:
    """``positions`` 单位是公共刻度（12 = 一拍）。

    逐窗口在两套格子里投票，再按选定的格子**绝对吸附**。

    ## 代价口径：为什么必须按密度归一化（``cost_normalization``）

    两套格的**格点数不同**：``step=3``（直十六分）每拍 4 个格点，``step=2``
    （三连音十六分）每拍 6 个。对拍内均匀分布的起音，到最近格点的平均距离有解析解
    ``step / 4``：

        直十六分  3/4 = 0.75 tick
        三连音    2/4 = 0.50 tick      ← 与音乐内容无关地白拿 0.25 tick

    于是直接比较两套格的**绝对**代价和，等于"谁格点多谁赢"：实测 40 万个均匀随机
    起音，三连音格在 **50.0%** 的起音上更近、直格只在 **16.6%** 上更近，
    平均代价差 ``+0.2504``。

    后果是**噪声会被读成三连音**：实测某首电子乐（无三连音证据）的代价差是
    ``+0.2546``，与纯随机的 ``+0.2504`` 几乎相同，但量化器照样输出 501 个
    全落在偶数 tick 上的"三连音"。同一把尺子也量过 victim（真三连音，92.2% 的起音
    落在偶数 tick），它拿到的 ``+0.2340`` **同样贴着随机基准**——答案对，过程错，
    所以这个缺陷长期没被发现。

    ``"density"``：各自除以 ``step / 4`` 的随机基准后再比（默认）。
    ``"raw"``：旧行为，保留用于对照与复现历史结果。

    ## 迟滞：``switch_margin``

    ``"density"`` 下阈值是**相对的**（每个起音的相对代价差），不再用 ``switch_penalty``
    那个绝对 tick 数。三连音是少数派假说，证据弱就沿用当前格子——某首曲子某个窗口
    的边距不足 1% 时，不会因为这一点点噪声就整段翻成三连音。

    ``merge_epsilon`` 只合并**近似同时**的起音（转谱抖动 / 一上一下的装饰），
    单位是公共刻度。**不能用"小于一个格步"做判据** —— 三连音十六分的相邻格点正好
    相差一个格步（2/12 拍），那样会把正常的连续音全并掉（实测会多并 705 个起音）。
    """
    if not positions:
        return Quantized([], "duple", 0.0, 0, 0.0, [])

    order = sorted(range(len(positions)), key=lambda i: positions[i])
    pos = [positions[i] for i in order]
    pit = [pitches[i] for i in order]

    # --- 1. 先合并近似同时的起音（音高取并集）------------------------------- #
    # 判据必须对着**簇首**比，不能对着"上一个"。对着上一个会**链式吞并**：
    # 只要每个相邻间隔都小于阈值，一整串连续起音会被并成一个（实测 yoake 因此
    # 从 1889 起音并到 797，而真正"小于 0.2 tick"的只有 245 个）。
    merged = 0
    cluster_start: list[float] = []
    cluster_pit: list[list[int]] = []
    for index, position in enumerate(pos):
        if cluster_start and position - cluster_start[-1] < merge_epsilon:
            merged += 1
            cluster_pit[-1] = sorted(set(cluster_pit[-1]) | set(pit[index]))
            continue
        cluster_start.append(position)
        cluster_pit.append(sorted(set(pit[index])))
    pos, pit = cluster_start, cluster_pit

    #: 每套格的"随机基准"：均匀分布下到最近格点的平均距离 = step / 4。
    #: 用它除，才把"格点密度"这个与音乐无关的因素除掉。
    baseline = {DUPLE: DUPLE / 4, TRIPLE: TRIPLE / 4}

    # --- 2. 逐窗口判定用哪套格子 ------------------------------------------- #
    choices: list[tuple[int, int, str]] = []      # (起, 止, 格子)
    current = "duple"
    start = 0
    while start < len(pos):
        end = min(start + window, len(pos))
        if end - start < 6:                       # 样本太少，沿用当前
            choices.append((start, len(pos), current))
            break
        chunk = pos[start:end]
        dup = sum(_error(p, DUPLE) for p in chunk)
        tri = sum(_error(p, TRIPLE) for p in chunk)
        if cost_normalization == "density":
            dup_c = dup / (len(chunk) * baseline[DUPLE])
            tri_c = tri / (len(chunk) * baseline[TRIPLE])
            # 相对的迟滞：只有比当前格子好过 switch_margin（每起音的相对代价差）才切换
            if current == "duple":
                chosen = "triple" if tri_c + switch_margin < dup_c else "duple"
            else:
                chosen = "duple" if dup_c + switch_margin < tri_c else "triple"
        else:
            # 旧行为：绝对 tick 代价 + 绝对 switch_penalty。保留用于对照。
            if current == "duple":
                chosen = "triple" if tri + switch_penalty < dup else "duple"
            else:
                chosen = "duple" if dup + switch_penalty < tri else "triple"
        choices.append((start, end, chosen))
        current = chosen
        if end >= len(pos):
            break
        start += hop

    # 把窗口结论铺到每个起音上（后写的窗口覆盖先写的重叠区）
    lattice_of = ["duple"] * len(pos)
    for begin, finish, name in choices:
        for index in range(begin, finish):
            lattice_of[index] = name

    # --- 2b. 连串一起吸附：不让单个起音把一段三连音切成切分 ---------------- #
    # 这是本轮的第二次修正。按窗口"逐起音各自决定格子"会留下一个具体失效：
    # 一段三连音里只要有个别起音被划到 duple，它就被吸到 3 tick 点上，前后间距
    # 从 2 变成 3 —— 用户听到的正是"三连音被写成切分"。
    # 实测（`run_best/diagnose_syncopation.py`）：victim 463 对三连音里 132 对被写成切分
    # （28.5%），最常见形态 `played 2.0 -> written 3.0`。
    #
    # 修法：先找出**成串**的三连音格起音（原始间隔接近 2 或 4 tick 的连续段），
    # 整串统一按 triple 吸附。判据只看原始间隔，不看四舍五入后的结果。
    run_start: int | None = None
    for index in range(len(pos)):
        if index == 0:
            run_start = 0
            continue
        gap = pos[index] - pos[index - 1]
        is_triplet_gap = abs(gap - 2.0) <= 0.9 or abs(gap - 4.0) <= 0.9
        if is_triplet_gap:
            if run_start is None:
                run_start = index - 1
            continue
        # 串结束
        if run_start is not None and index - run_start >= 2:
            for member in range(run_start, index):
                lattice_of[member] = "triple"
        run_start = None
    if run_start is not None and len(pos) - run_start >= 2:
        for member in range(run_start, len(pos)):
            lattice_of[member] = "triple"

    # --- 3. 绝对吸附到选定的格子 ------------------------------------------- #
    #
    # `snap_policy` 决定"某个起音最终落在哪个格点"：
    #
    #   "family"（旧行为）：一律吸到**本窗口选定的族**的格点上。族一旦判成 triple，
    #       窗口内所有起音都被吸到偶数 tick —— 哪怕某个起音离直格点更近。
    #       实测这种"族承诺强迫"在 yoake 上挪远了 11.6% 的起音、某首电子乐上
    #       挪远了 **15.6%**，平均位移反而比"就近"更大（0.491 vs 0.420 tick）。
    #       族承诺买到的是节奏族干净，代价是把音放到离原始演奏更远的位置。
    #
    #   "nearest"：每个起音在两族的格点里挑**离它真实位置更近**的那个，
    #       即使这违反窗口的族判定。被挪远的起音归零、总位移最小，
    #       并且不会把"本来在 3 tick 附近"的音硬挪到 2 或 4（那正是
    #       "直十六分被写成三连音"的来源）。代价是可能产出混族，
    #       靠雕版阶段的 `regularize_tuplet_values` 收拾落单的三连音。
    #
    #   默认仍是 "family"：换成 "nearest" 会改变所有已验收样例的谱面，
    #   需要听感确认后再切。
    snapped: list[int] = []
    for index, position in enumerate(pos):
        step = DUPLE if lattice_of[index] == "duple" else TRIPLE
        # 先在"拍"层面取整再在拍内吸附，避免跨拍累积漂移
        beat = int(position // COMMON)
        within = position - beat * COMMON
        chosen = beat * COMMON + _nearest_on_lattice(within, step)
        if snap_policy == "nearest":
            other = TRIPLE if step == DUPLE else DUPLE
            alternative = beat * COMMON + _nearest_on_lattice(within, other)
            # 用**严格更近**才改判，避免平局时来回抖
            if abs(position - alternative) < abs(position - chosen) - 1e-9:
                chosen = alternative
        snapped.append(chosen)

    # --- 4. 吸附到同一格点的再合并（音高并集）------------------------------- #
    onsets: list[tuple[int, list[int]]] = []
    for index, tick in enumerate(snapped):
        if onsets and tick == onsets[-1][0]:
            merged += 1
            onsets[-1] = (tick, sorted(set(onsets[-1][1]) | set(pit[index])))
            continue
        onsets.append((tick, list(pit[index])))

    max_shift = max(
        (abs(pos[index] - snapped[index]) for index in range(len(pos))),
        default=0.0,
    )

    duple_count = sum(1 for name in lattice_of if name == "duple")
    triplet_count = len(lattice_of) - duple_count
    share = duple_count / len(lattice_of)
    if duple_count and triplet_count:
        kind = "mixed"
    else:
        kind = "duple" if duple_count else "triple"

    return Quantized(
        onsets=onsets,
        lattice=kind,
        duple_share=share,
        merged=merged,
        max_shift_ticks=round(max_shift, 3),
        per_window=[name for _b, _f, name in choices],
    )
