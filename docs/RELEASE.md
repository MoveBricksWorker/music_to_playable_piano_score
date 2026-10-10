# 发布与维护清单

这份清单记录**发布这个仓库时做了什么、还需要你手动补什么**。

---

## 一、仓库里跟踪了什么 / 忽略什么

`.gitignore` 的规则，以及为什么：

| 目录 | 大小 | 是否跟踪 | 为什么 |
|---|---|---|---|
| `pipeline/`、`ensemble2score/`、根目录脚本 | ≈ 1 MB | ✅ 跟踪 | 这就是代码本体 |
| `out/**/*.{musicxml,mid,pdf}`、`engrave.json`、`report.json` | ≈ 5 MB | ✅ 跟踪 | 四首成品样例，别人不出谱就能看效果 |
| `out/**/arrangement.txt`、`*.melody.json` | ≈ 250 KB | ❌ 忽略 | 工作中间产物，可由 `--from arrange` 重新生成 |
| `python/` | 6.4 GB | ❌ 忽略 | 第三方解释器与依赖，走网盘 |
| `models/` | 570 MB | ❌ 忽略 | 模型权重（其中一个还是非商业许可），走网盘 |
| `data/` | 362 MB | ❌ 忽略 | 输入音频与中间产物，走网盘 |
| `work/` | 530 MB | ❌ 忽略 | 工作区，`seed_work.py` 可从 `data/` 重建 |
| `_out_shape_check.txt` 等运行报告 | 小 | ❌ 忽略 | 每次跑都会变 |

**总跟踪体积约 5 MB / 104 个文件**，克隆很快。

> 如果你**确实**想把大目录也放进版本库，请用 Git LFS，不要直接 `git add`：
> ```bash
> git lfs track "*.wav" "*.safetensors" "*.th" "*.dll" "*.pt"
> ```
> 但免费账户的 LFS 配额（1 GB 存储 / 1 GB 月流量）放不下这 8 GB，
> 网盘才是对的方案。

> **关于 GitHub Release 能不能放整合包**：单文件上限是 2 GiB，整合包里最大的
> 单文件是 913 MB（`torch_cuda.dll`），所以**技术上拆成分卷可以传**。但不建议：
> Release 是用来分发软件版本的，把 8 GB 训练环境镜像流量挂上去，既容易触到
> 合理使用政策，也让仓库变成一个不好维护的存储桶。Release 的正文里放网盘
> 链接即可。

---

## 二、发布前必须补的占位符

仓库名已定为 **`music_to_playable_piano_score`**、Owner 为 **`MoveBricksWorker`**，
下列项已完成：

- [x] ~~README 里的网盘链接~~ —— 已填夸克网盘链接与提取码 `Wuqn`
      （`README.md` 的「方式 A：用整合包」一节）。
- [x] ~~README 顶部的 CI 徽章~~ —— 已填 `MoveBricksWorker`（徽章 URL 出现两次）。
- [x] ~~`pyproject.toml` 的 name~~ —— 已改为 `music-to-playable-piano-score`。

- [x] ~~**git 身份**~~ —— 已配 `MoveBricksWorker <2673992653@qq.com>`（仓库级）。
      换机器或换仓库时需重配，见第三节：不配则 `git commit` 直接失败。

仍需处理：

- [ ] **LICENSE 里的版权人**：把 `<你的名字或 GitHub ID>` 换成你希望公开显示的名字。
- [ ] 检查 `docs/images/victim-page2.png` 是否愿意公开（那是样例曲的输出）。

自检命令：

```bash
grep -rn "待填\|你的网盘链接\|你的名字\|你的ID\|TODO(发布前)" --include="*.md" --include="*.py" --include="*.toml" .
```

---

## 三、首次提交

> ⚠️ **本机目前没有配 git 身份，第一次 `git commit` 会直接失败。**
> 实测：`git var GIT_COMMITTER_IDENT` 返回 exit 128，git 只能从主机名猜出一个
> 无效邮箱（`<用户名>@<主机名>.(none)` 这种形式），拿它提交会被拒。
> （`git commit --dry-run` 不检查身份，所以它会"看起来成功"，别被骗。）
>
> 下面两条**必须**先跑。只想让这个仓库认这个身份，就去掉 `--global`
> （`user.email` 用 GitHub 帐号里的邮箱，或那个 `…@users.noreply.github.com` 形式）。

```bash
git config user.name "MoveBricksWorker"
git config user.email "you@example.com"

# 验证：应当打印出 "MoveBricksWorker <you@example.com>"
git var GIT_COMMITTER_IDENT
```

然后：

```bash
cd music_to_playable_piano_score

git add -A
git status --short | head -30        # 确认没有 python/ models/ data/ work/
git commit -m "初始提交：音频转可弹钢琴谱管线

九个阶段：Demucs 分离 → 相加 → Transkun 转谱 → SheetSage2 拍点/调号/旋律
→ 12 tick/拍双格量化编排 → 可弹性修正 → 雕版 → MuseScore 渲染 → 谱面验收。

整合包（自带解释器与模型）走网盘，仓库只放代码、文档与四首样例成品。"
git branch -M main
```

提交前请再确认一次：

```bash
git ls-files | wc -l                       # 期望 100 上下
git ls-files | grep -E "^(python|models|data|work)/"   # 期望：无输出
du -sh .git                                # 期望：几 MB
```

---

## 四、推到 GitHub

仓库已在网页上建好（`MoveBricksWorker/music_to_playable_piano_score`，**空仓库、不带
README / .gitignore / LICENSE**，否则首推会冲突），然后：

```bash
git remote add origin https://github.com/MoveBricksWorker/music_to_playable_piano_score.git
git push -u origin main
```

> 若不小心在网页上勾了 README 或 LICENSE，首推会被拒（远端有本地没有的提交）。
> 处理办法：`git pull --rebase origin main` 把那个自动提交并进来，再推。

### 仓库设置建议

- **Description**（已选定，可直接粘贴）：

  ```
  Audio → playable piano sheet music (MusicXML/MIDI/PDF): Demucs + Transkun + SheetSage2, 12-tick dual-lattice quantization, hand-span/playability constraints, score-level verification.
  ```

- **Topics**（每个都点得进去，是真实流量入口）：

  `music-transcription` `piano` `musicxml` `audio-to-score` `quantization`
  `sheet-music` `music21` `demucs` `transkun` `sheetsage2`

- **Releases**：把整合包链接挂到 Release 描述里，比只写在 README 里更容易被找到。
  建议打一个 `v1.0.0` 标签。
- **Issues**：建议至少开三个 issue 记着 README「已知缺陷」里的头三条
  （临时记号过多、旋律保护顶掉约束、缺真值评测），这样别人知道你在意它们。

---

## 五、GitHub 上的大文件红线

单文件 **100 MB** 是硬限制，超过直接拒绝推送；单文件 >50 MB 会警告。
本仓库跟踪的文件最大约 1 MB（`out/victim/victim.musicxml`），远低于限制。

**千万不要**为了让整合包进仓库而 `git add -f python/`：那会是 10 万个文件、
6.4 GB，推送会失败，而且清理历史很麻烦。如果不小心提交了，用
`git rm -r --cached python/` 撤回并重新提交。

---

## 六、后续维护待办（按价值排序）

1. **修临时记号过多**——`_pitch_with_spelling` 不要显式带 accidental，交给
   `music21.makeAccidentals` 按调号推导。四首能少掉约 500–1200 个多余记号。
   见 README「已知缺陷」。
2. **修 R3/R4**——把 `[1/8, 1/2]` 拍的时值夹子下沉进 `normalize_durations`，
   或调整它与 `regularize_tuplet_values` 的先后。这条风险最小、收益最直接。
3. **把 `ties` 统计接对**——`build_score_exact` 路径没有给 `stats.ties` 赋值，
   于是 `engrave.json` 永远写 0，而实际有 50–111 条连音线。
4. **保持 CI 绿色**——`.github/workflows/verify.yml` 已经在守三条硬指标
   （宽和弦 0 / 拥挤起音 0 / R1 = 0 / 不丢音 / 有调号）+ 仓库卫生（无大文件、无大目录）。
   往 `tests/regression.py` 加断言前，先确认那条指标真的达标了。
5. **确认编辑器读了 `.editorconfig`**——仓库里出现过的乱码（产物 JSON 里的中文说明）
   就是没统一编码的结果；文件已经加上了，剩下的是习惯问题。
6. **清理历史脚本**——`pipeline/build_sum.py` 已经没人调用，而且里面的路径
   （`parents[2]` + `deliverables_4stem/`）还是上游项目的。要么删掉，
   要么改成和 `run_pipeline.stage_sum` 共用一份实现。
7. **Linux/macOS 验证**——`install_env.py` 有 POSIX 分支，但没人跑过。
   CI 现在能在 Linux 上跑验收链路，但建环境那一步没进 CI。
8. **中段转调发调号**——`material.key_changes` 已经读进来了，
   但 `_key_signature_for` 只在开头调用一次。见 README「已知缺陷」。
