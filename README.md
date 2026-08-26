# ListenNote - 系统音频语音识别

两种模式：实时捕获系统音频转文字，或将已有音频文件转文字。

识别基于 **FunASR 模型(paraformer/sensevoice)+ VAD 断句 + 标点恢复**，自动按句子分段并加标点，适合提取视频文案、会议记录等。

## 快速开始

1. 双击 `install.bat` 创建虚拟环境并安装依赖
2. 双击 `run.bat`，在菜单里选模式：

```
  [1] 录音+自动转写(先录,停止后整段转写) — 推荐
  [2] 实时字幕(边看边出字)
  [3] 批量转写(重新扫描 input/)
```

也可以命令行直接指定：

```bash
run.bat --file lecture.mp3       # 转写指定文件(已有转写会跳过)
run.bat --file lecture.mp3 --force  # 强制重新转写
run.bat --live                   # 直接进实时字幕
run.bat --record                 # 直接开始录音
```

识别结果保存在 `output/` 目录

## 常驻识别服务（启动快的关键）

模型加载一次约 15~30 秒。程序会在**第一次启动时自动在后台拉起常驻服务**，之后每次启动都**秒连**（约 2~3 秒），不再重复加载模型。

- 手动启停：
  ```bash
  python ListenNote.py --server        # 前台运行服务
  python ListenNote.py --stop-server   # 停止后台服务
  ```
- 服务空闲超过 `server_idle_exit_minutes` 自动退出（默认 120 分钟）
- 服务日志在 `output/server.log`
- 服务异常/超时时自动回退为进程内加载模型，不影响使用

## 纯录制模式（一个接一个录，窗口常驻，推荐）

看多个视频时，录制只管录制，最后一次性转写：

1. 双击 `run.bat`（无录音任务时默认就是录音模式）→ **窗口常驻**，开始录第一个视频
2. 播放视频/音频，完成后**按任意键停止**（不要按 Ctrl+C）
3. 提示「按任意键录下一个 | 按 T 转写当前 | 按 Q 退出」→ 继续录下一个
4. 录完后按 Q 退出，再用 `run.bat` 选批量转写，一次性转完所有录音

每个视频存成独立的 `input/record_时间.wav`（16kHz 单声道，比 48k 立体声小约 6 倍）。批量转写会自动跳过已转过的，只转新的。

## 实时模式（识别更准）

- 按 `chunk_seconds`（默认 8s）提交识别窗口，模型内部 **VAD 断句 + 标点**，输出的是完整带标点的句子，而不是被切碎的小段
- **全程录音**：实时识别的同时把整个会话存为 `session_*.wav`
- **结束时自动整段重识别**：按 Ctrl+C 或空闲超时停止后，会询问是否整段重识别——对全会话音频重新做一次断句级转写，得到最准确的最终文案（20 分钟音频约需十几秒）

## 文件转录模式（批量，自动跳过已转写）

所有音频都放 `input/`（录制模式也自动存到这里）。双击 `run.bat` 菜单选批量转写会**扫描 `input/`，自动跳过已有转写**的文件，只转新的。

支持格式：wav、flac、ogg、mp3、m4a、aac、wma、opus（mp3/m4a 需要 ffmpeg）。

```bash
run.bat                            # 出菜单,选批量转写:扫描 input/,跳过已转的
run.bat --file x.wav               # 转写指定文件(已有转写会跳过)
run.bat --file x.wav --force       # 强制重新转写
```

识别结果保存在 `output/<文件名>_transcribed.txt`

## A/B 说话人分人

两人对话的视频/录音，把 `config.json` 里 `diarize` 设为 `1`（自动判断人数）或 `2`（固定两人），转写结果会自动给每句加上 `[A]` / `[B]` 说话人标签：

```
[A] 你觉得这个方案怎么样？
[B] 我觉得可以，就是成本有点高。
[A] 那我们先按这个思路试试。
```

- **`diarize: 1` 自动判断人数**（推荐，不用自己数）；`diarize: 2` 固定按两人分
- **需要 `paraformer` 模型**（SenseVoice / Whisper 不支持分人）
- 开启会额外加载一个声纹模型（磁盘 28MB、内存约 200MB）
- **粗略版**：实际只有 1 人说话时可能被错分；3 人以上对话会把多人合并成两组
- 分人转写出错时自动退回普通转写（不丢文案）

## 模型对比

| 模型 | 语言 | 质量 | 速度 | 说明 |
|------|------|------|------|------|
| Paraformer | 仅中文 | 高 | 快 | 阿里达摩院 large 版，纯中文无多语种残留，**中文推荐** |
| SenseVoice | 多语言 | 最高 | 快 | 阿里最新模型，支持情感，但中文输出偶有日文假名残留 |
| Whisper | 多语言 | 中 | 较慢 | OpenAI，中文效果一般，适合英文 |

## 配置说明 (config.json)

首次运行自动生成，所有参数均可直接修改，无需改代码。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `model_backend` | `"paraformer"` | `"paraformer"`(中文推荐) / `"sensevoice"`(多语种) / `"whisper"` |
| `whisper_size` | `"small"` | Whisper 专用：`tiny` / `base` / `small` / `medium` / `large-v3` |
| `language` | `"zh"` | 识别语言：`"zh"` 中文 / `"en"` 英文 |
| `chunk_seconds` | `8` | 实时识别窗口秒数：3=低延迟 5=均衡 8=句子更完整 10=高准确率 |
| `idle_timeout` | `300` | 无声音自动退出秒数，`0` 禁用 |
| `server_port` | `17777` | 常驻识别服务端口 |
| `server_idle_exit_minutes` | `120` | 服务空闲自动退出分钟数，`0` 常驻 |
| `diarize` | `1` | 说话人分人：`0` 不分 / `1` 自动判断人数(推荐) / `2` 固定两人分 A/B(需 paraformer) |

### 配置示例

**中文会议记录（高准确率）：**
```json
{
  "model_backend": "paraformer",
  "language": "zh",
  "chunk_seconds": 10,
  "idle_timeout": 600,
  "server_idle_exit_minutes": 0
}
```

**英文实时字幕（低延迟）：**
```json
{
  "model_backend": "whisper",
  "whisper_size": "small",
  "language": "en",
  "chunk_seconds": 3,
  "idle_timeout": 0
}
```

## 离线可用（无需联网）

模型下载一次后会缓存在 `%USERPROFILE%\.cache\modelscope`。程序启动时**自动检测本地缓存，命中就直接离线加载**，不再联网解析——断网、代理关闭都不会卡住转写。只有首次下载模型时才需要网络（建议开代理/VPN）。

## 项目结构

```
ListenNote/
├── ListenNote.py       # 主程序（含常驻服务）
├── config.json         # 配置文件（自动生成）
├── input/              # 放入待转录的音频文件
├── output/             # 生成的识别文件（captions_*.txt / *_transcribed.txt / server.log）
├── install.bat         # 安装依赖脚本
├── run.bat             # 统一入口：菜单选 录音+自动转写 / 实时字幕 / 批量转写
├── tests/              # 单元测试（python -m unittest tests.test_listennote）
├── requirements.txt    # Python 依赖
└── README.md
```

## 许可证

MIT
