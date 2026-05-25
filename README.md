# ListenNote - 系统音频语音识别

两种模式：实时捕获系统音频转文字，或将已有音频文件转文字。

## 快速开始

1. 双击 `install.bat` 创建虚拟环境并安装依赖
2. 选择启动方式：

| 脚本 | 说明 |
|------|------|
| `run-file.bat` | 文件转录模式，扫描 `input/` 中的音频文件 |
| `run-live.bat` | 实时系统音频识别模式 |

也可以命令行指定文件：

```bash
run-file.bat --file lecture.mp3
```

识别结果保存在 `output/` 目录

## 文件转录模式

把音频文件放入 `input/` 文件夹，启动后选择「转录文件」即可逐个处理。

支持格式：wav、flac、ogg、mp3、m4a、aac、wma、opus（mp3/m4a 需要 ffmpeg）。

```bash
run.bat                    # 自动检测 input/ 中的文件，选择模式
run.bat --file lecture.mp3 # 直接转录指定文件
```

识别结果保存在 `output/<文件名>_transcribed.txt`

## 模型对比

| 模型 | 语言 | 质量 | 速度 | 说明 |
|------|------|------|------|------|
| Paraformer | 仅中文 | 高 | 快 | 阿里达摩院，自带 VAD + 标点恢复 |
| SenseVoice | 多语言 | 最高 | 快 | 阿里最新模型，支持情感识别 |
| Whisper | 多语言 | 中 | 较慢 | OpenAI，中文效果一般，适合英文 |

## 配置说明 (config.json)

首次运行自动生成，所有参数均可直接修改，无需改代码。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `model_backend` | 首次选择 | `"paraformer"` / `"sensevoice"` / `"whisper"` |
| `whisper_size` | `"small"` | Whisper 专用：`tiny` / `base` / `small` / `medium` / `large-v3` |
| `language` | `"zh"` | 识别语言：`"zh"` 中文 / `"en"` 英文 |
| `chunk_seconds` | `5` | 识别间隔秒数：3=低延迟 5=均衡 10=高准确率 |
| `idle_timeout` | `300` | 无声音自动退出秒数，`0` 禁用 |

### 配置示例

**中文会议记录（高准确率）：**
```json
{
  "model_backend": "paraformer",
  "language": "zh",
  "chunk_seconds": 10,
  "idle_timeout": 600
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

## 项目结构

```
ListenNote/
├── ListenNote.py       # 主程序
├── config.json         # 配置文件（自动生成）
├── input/              # 放入待转录的音频文件
├── output/             # 生成的识别文件
├── install.bat         # 安装依赖脚本
├── run-file.bat        # 文件转录模式
├── run-live.bat        # 实时系统音频模式
├── requirements.txt    # Python 依赖
└── README.md
```

## 许可证

MIT
