"""精简版最优管线：音频 → 可弹钢琴谱。

模块划分：

    arrange            编排（12 tick/拍 双格量化、中音C分手、旋律最高声部）
    engrave            雕版（时值规整、小节填满、按调号拼写、速度标记）
    dual_lattice       双格量化器
    fix_playable       可弹性修正（跨度 ≤19 半音、同时音 ≤5）
    verify_playable    可弹性验收
    check_shapes       谱面形状验收（R1–R4）

上游阶段的脚本（分离 / 相加 / 转谱 / SheetSage2）在 `pipeline/` 下同名文件里，
由 `run_pipeline.py` 以子进程方式调用。
"""

from __future__ import annotations

__all__ = ["arrange", "engrave", "dual_lattice", "fix_playable",
           "verify_playable", "check_shapes"]
