# ListenNote - 系统音频实时语音识别

捕获电脑正在播放的所有声音（视频、音乐、会议等），实时识别语音并保存为文本文件。

## 功能

- 通过 WASAPI 环回捕获系统所有音频输出（Windows）
- 使用 faster-whisper 进行实时语音识别
- 支持中文及多种语言
- 每次运行自动保存到带时间戳的独立文件（`output/` 目录）
- 自动跳过静音段，避免产生乱码

## 环境要求

- Windows 10/11
- Python 3.8+
- 需要连接扬声器或耳机（用于环回捕获）

## 快速开始

1. 双击 `install.bat` 创建虚拟环境并安装依赖
2. 在电脑上播放任意视频/音频
3. 双击 `run.bat` 开始识别
4. 按 `Ctrl+C` 停止

识别结果保存在 `output/captions_YYYYMMDD_HHMMSS.txt`

## 配置说明

编辑 `ListenNote.py` 顶部的配置项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SECONDS` | 15 | 每次识别间隔秒数（越短越实时，越长越准确） |
| `MODEL_SIZE` | `medium` | Whisper 模型：`tiny` / `base` / `small` / `medium` / `large-v3` |
| `DEVICE` | `cpu` | `cpu` 或 `cuda`（NVIDIA 显卡加速） |
| `LANGUAGE` | `zh` | 语言代码（`zh` 中文，`en` 英文等） |

## 模型大小参考

| 模型 | 大小 | 速度 | 质量 |
|------|------|------|------|
| tiny | ~75MB | 最快 | 较低 |
| base | ~150MB | 快 | 基础 |
| small | ~500MB | 中等 | 良好 |
| medium | ~1.5GB | 较慢 | 优秀（推荐中文使用） |
| large-v3 | ~3GB | 最慢 | 最佳 |

## 项目结构

```
ListenNote/
├── ListenNote.py       # 主程序
├── install.bat         # 安装依赖脚本
├── run.bat             # 启动脚本
├── requirements.txt    # Python 依赖
├── output/             # 生成的识别文件
└── README.md
```

## 许可证

MIT
