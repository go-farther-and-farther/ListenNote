import os
# 必须在 numpy/torch 导入前设置,避免 MKL/OpenMP 线程在进程退出时报
# "forrtl: error (200): program aborting due to window-CLOSE event" 崩溃
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import re
import numpy as np
import threading
import time
import wave
import tempfile
import json
import logging
import urllib.request
from datetime import datetime

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_DISABLE_PROGRESS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import pyaudiowpatch as pyaudio

# ========== 默认配置 ==========
DEFAULT_CHUNK_SECONDS = 8       # 默认识别间隔：3s=低延迟 5s=均衡 10s=高准确率
DEFAULT_IDLE_TIMEOUT = 300      # 无声音自动退出秒数，0=禁用
DEFAULT_LANGUAGE = "zh"         # 默认识别语言：zh=中文 en=英文
DEFAULT_DIARIZE = 0             # 说话人分人：0=不分 1=自动判断 2=固定两人
DEFAULT_SERVER_PORT = 17777     # 常驻识别服务端口
DEFAULT_SERVER_IDLE_MIN = 120   # 常驻服务空闲自动退出分钟，0=常驻
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".wma", ".opus"}

DEVICE = "cpu"                  # cpu / cuda(N卡加速)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# 间隔选择提示
CHUNK_HINT = {
    3: "低延迟模式，适合实时对话",
    5: "均衡模式，兼顾实时性与准确率",
    8: "句子更完整模式（推荐）",
    10: "高准确率模式，适合会议记录",
}

def _check_pkg(pkg):
    """检测某个包是否已安装"""
    from importlib.util import find_spec
    return find_spec(pkg) is not None

def check_models_status():
    """检测各模型依赖的安装状态，返回 {(status, hint), ...}"""
    has_funasr = _check_pkg("funasr")
    has_faster_whisper = _check_pkg("faster_whisper")
    return {
        "1": has_faster_whisper,
        "2": has_funasr,
        "3": has_funasr,
    }

MODEL_OPTIONS = {
    "1": {
        "backend": "whisper",
        "name": "Whisper (多语言)",
        "desc": "OpenAI Whisper，支持多语言，中文效果一般",
        "pip": "faster-whisper",
        "sizes": {"tiny": "~75MB", "base": "~150MB", "small": "~500MB", "medium": "~1.5GB", "large-v3": "~3GB"},
    },
    "2": {
        "backend": "paraformer",
        "name": "Paraformer (中文专用)",
        "desc": "阿里达摩院中文专用模型，识别准确率高，自带标点恢复",
        "pip": "funasr modelscope torch torchaudio onnxruntime",
    },
    "3": {
        "backend": "sensevoice",
        "name": "SenseVoice (中文最佳)",
        "desc": "阿里最新中文语音模型，效果最好，支持情感识别",
        "pip": "funasr modelscope torch torchaudio onnxruntime",
    },
}
# ===========================


# ========== 配置管理 ==========
def _strip_json_comments(text):
    """去除 JSON 中的 // 和 # 注释"""
    lines = []
    for line in text.split("\n"):
        # 去掉行尾注释（但不影响字符串内的内容）
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        # 去掉行内注释(仅当 // 位于行首或前面是空白,避免误截字符串内的 URL 等)
        in_string = False
        for i, ch in enumerate(line):
            if ch == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
            if (not in_string and line[i:i+2] == "//"
                    and (i == 0 or line[i-1].isspace())):
                line = line[:i]
                break
        lines.append(line)
    return "\n".join(lines)


def load_config():
    """从 config.json 加载配置，自动补全缺失字段"""
    defaults = {
        "language": DEFAULT_LANGUAGE,
        "chunk_seconds": DEFAULT_CHUNK_SECONDS,
        "idle_timeout": DEFAULT_IDLE_TIMEOUT,
        "diarize": DEFAULT_DIARIZE,
        "server_port": DEFAULT_SERVER_PORT,
        "server_idle_exit_minutes": DEFAULT_SERVER_IDLE_MIN,
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        try:
            config = json.loads(_strip_json_comments(text))
        except json.JSONDecodeError as e:
            print(f"[错误] config.json 解析失败: {e}")
            # 备份损坏文件,避免覆盖用户手写内容
            import shutil
            try:
                shutil.copy2(CONFIG_FILE, CONFIG_FILE + ".bak")
                print(f"  已备份原文件到 {CONFIG_FILE}.bak")
            except Exception:
                pass
            config = {}
        # 内存中补全缺失字段;不写回文件,以免抹掉用户手写的注释
        for key, val in defaults.items():
            config.setdefault(key, val)
        return config
    return {}


def save_config(config):
    """保存配置到 config.json"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def merge_selection_config(existing, backend, language, whisper_size=None):
    """把新的模型选择结果合并进已有配置:只更新 model/language(及 Whisper 大小),
    保留 diarize/server_port/idle_timeout 等用户自定义字段,缺失的补默认值。"""
    cfg = dict(existing or {})
    cfg["model_backend"] = backend
    cfg["language"] = language
    if whisper_size is not None:
        cfg["whisper_size"] = whisper_size
    for key, val in {
        "chunk_seconds": DEFAULT_CHUNK_SECONDS,
        "idle_timeout": DEFAULT_IDLE_TIMEOUT,
        "diarize": DEFAULT_DIARIZE,
        "server_port": DEFAULT_SERVER_PORT,
        "server_idle_exit_minutes": DEFAULT_SERVER_IDLE_MIN,
    }.items():
        cfg.setdefault(key, val)
    return cfg


def prompt_model_selection():
    """每次启动选择语言，再自动匹配模型"""
    # 1. 先选语言
    print("=" * 55)
    print("  选择识别语言 / Select language：")
    print("=" * 55)
    print("  [1] 中文 (zh)")
    print("  [2] English (en)")
    while True:
        lang_choice = input("\n请输入编号 (1/2) [1]: ").strip() or "1"
        if lang_choice in ("1", "2"):
            break
        print("无效输入，请重新选择")
    language = "zh" if lang_choice == "1" else "en"

    installed = check_models_status()

    # 2. 英文 → 自动选 Whisper；中文 → 手动选模型
    if language == "en":
        if installed["1"]:
            print("\n  英文识别自动使用 Whisper 模型")
        else:
            print(f"\n  [ ! ] Whisper 依赖未安装，请运行：pip install faster-whisper")
        selected = MODEL_OPTIONS["1"]  # Whisper
    else:
        print(f"\n{'=' * 55}")
        print("  选择语音识别模型：")
        print("=" * 55)
        for key, opt in MODEL_OPTIONS.items():
            tag = "已安装" if installed[key] else "未安装"
            icon = "+" if installed[key] else "x"
            print(f"  [{key}] [{icon}] {opt['name']} — {opt['desc']}")

        while True:
            choice = input("\n请输入编号 (1/2/3) [2]: ").strip() or "2"
            if choice in MODEL_OPTIONS:
                break
            print("无效输入，请重新选择")
        selected = MODEL_OPTIONS[choice]
        if not installed[choice]:
            print(f"\n  [ ! ] {selected['name']} 依赖未安装，请运行：pip install {selected['pip']}")

    # 与已有配置合并,不覆盖用户自定义字段(diarize/server_port 等)
    config = merge_selection_config(load_config(), selected["backend"], language)

    # Whisper 额外选择模型大小
    if selected["backend"] == "whisper":
        print(f"\n可选模型大小：")
        for size, mem in selected["sizes"].items():
            print(f"  {size:10s} {mem}")
        while True:
            size = input("\n请输入模型大小 [small]: ").strip() or "small"
            if size in selected["sizes"]:
                break
            print("无效输入，请重新选择")
        config["whisper_size"] = size

    save_config(config)
    lang_disp = {"zh": "中文", "en": "English"}[language]
    print(f"\n  {lang_disp} | {selected['name']} → 已保存到 {CONFIG_FILE}\n")
    return config


# ========== 模型加载 ==========
MODELSCOPE_HUB_DIR = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models")

def _local_model_dir(model_id):
    """若模型已完整缓存在本地 modelscope hub,返回其绝对路径(离线加载)。
    FunASR 传入模型 ID 时每次启动都会联网向 modelscope 解析快照——断网/代理关闭时
    会永久卡死;传本地目录则完全不联网。缓存不完整(缺 configuration.json)或未缓存
    时原样返回模型 ID(首次下载场景)。"""
    local = os.path.join(MODELSCOPE_HUB_DIR, *model_id.split("/"))
    if os.path.isfile(os.path.join(local, "configuration.json")):
        return local
    return model_id

def load_whisper_model(size):
    """加载 Whisper 模型"""
    from faster_whisper import WhisperModel
    print(f"正在加载 Whisper {size} 模型...")
    model = WhisperModel(
        size,
        device=DEVICE,
        compute_type="int8" if DEVICE == "cpu" else "float16",
    )
    print("OK 模型加载完成\n")
    return model


def load_funasr_model(model_name, **kwargs):
    """加载 FunASR 模型（Paraformer / SenseVoice）"""
    import sys
    print(f"正在加载 {model_name} 模型...")
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        logging.getLogger("modelscope").setLevel(logging.ERROR)
        logging.getLogger("funasr").setLevel(logging.ERROR)
        from funasr import AutoModel
        model = AutoModel(model=model_name, disable_update=True, **kwargs)
    except ImportError:
        print(f"[错误] 未安装 funasr，请运行：")
        print(f"  pip install funasr modelscope torch torchaudio onnxruntime")
        raise
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    print(f"OK 模型加载完成\n")
    return model


def _funasr_extra_kwargs(backend, diarize):
    """FunASR 模型加载的额外参数:A/B 分人需加载声纹模型(cam++)。
    只有 paraformer 能输出时间戳并支持分人,其他后端不传 spk_model,
    避免白下载/白占内存。"""
    if int(diarize or 0) <= 0:
        return {}
    if backend == "paraformer":
        return {"spk_model": _local_model_dir("iic/speech_campplus_sv_zh-cn_16k-common")}
    print(f"[提示] A/B 分人需要 paraformer 模型(能输出时间戳),当前为 {backend},分人将不生效")
    return {}


def load_model(config):
    """根据配置加载对应模型，返回 (model, backend)"""
    backend = config.get("model_backend", "whisper")
    spk_kw = _funasr_extra_kwargs(backend, int(config.get("diarize", 0) or 0))

    if backend == "whisper":
        size = config.get("whisper_size", "small")
        model = load_whisper_model(size)
        return model, "whisper"

    elif backend == "paraformer":
        model = load_funasr_model(
            _local_model_dir("iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
            vad_model=_local_model_dir("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
            punc_model=_local_model_dir("iic/punc_ct-transformer_cn-en-common-vocab471067-large"),
            **spk_kw,
        )
        return model, "paraformer"

    elif backend == "sensevoice":
        model = load_funasr_model(
            _local_model_dir("iic/SenseVoiceSmall"),
            trust_remote_code=True,
            vad_model=_local_model_dir("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
            punc_model=_local_model_dir("iic/punc_ct-transformer_cn-en-common-vocab471067-large"),
            **spk_kw,
        )
        return model, "sensevoice"

    else:
        raise ValueError(f"未知的模型后端: {backend}")


# ========== 统一转录接口 ==========
def transcribe_whisper(model, wav_path, language):
    """Whisper 转录"""
    initial_prompt = "以下是普通话的句子，使用简体中文输出。" if language == "zh" else ""
    segments, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        initial_prompt=initial_prompt,
    )
    texts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            texts.append(text)
    return texts


def _recognize_chunk(model, chunk):
    """识别单个音频分段（供线程池调用）"""
    results = model.generate(input=chunk, input_len=len(chunk))
    texts = []
    for result in results:
        text = result.get("text", "").strip()
        if text:
            # 去掉 SenseVoice 特殊标记 <|zh|> <|NEUTRAL|> 等：
            # VAD 分段后这些标记会出现在段间(不只开头),且内部可能含空格
            text = re.sub(r'<\s*\|[^>]*\|\s*>', '', text).strip()
            if text:
                texts.append(text)
    return texts


def _collect_chunk_texts(chunks, results_map):
    """按 chunk 实际顺序收集识别结果(chunks: [(idx, array), ...])。
    某段被跳过时后续 chunk 的 idx 不连续,按 range(len(chunks)) 收集会丢文本。"""
    texts = []
    for idx, _chunk in chunks:
        texts.extend(results_map.get(idx, []))
    return texts


def transcribe_funasr(model, wav_path, quiet=False):
    """FunASR 转录（Paraformer / SenseVoice），顺序识别。quiet=True 时静默(实时/服务端用)"""
    import soundfile as sf

    # 显示音频信息
    try:
        info = sf.info(wav_path)
        duration = info.duration
        sr = info.samplerate
        if not quiet:
            print(f"    音频时长: {duration/60:.1f} 分钟, 采样率: {sr} Hz")
    except Exception as e:
        print(f"    [错误] 无法读取音频文件: {e}")
        return []

    # 读取所有分段（内存可控：每分钟 16kHz float32 ≈ 3.8MB）
    target_sr = 16000
    need_resample = sr != target_sr
    if need_resample:
        import torch
        import torchaudio.functional as F

    # 3 分钟一段：配合模型内部 VAD 自动按句子断句，避免句子被固定边界切断
    chunk_seconds = 180
    chunk_samples = sr * chunk_seconds
    total_frames = info.frames
    num_chunks = max(1, (total_frames + chunk_samples - 1) // chunk_samples)

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_samples
        frames = min(chunk_samples, total_frames - start)
        if frames < sr:
            continue
        chunk, _ = sf.read(wav_path, dtype="float32", start=start, frames=frames)
        if chunk.ndim > 1:
            chunk = chunk[:, 0]
        if need_resample:
            chunk = F.resample(torch.from_numpy(chunk), sr, target_sr).numpy()
        chunks.append((i, chunk))

    # 顺序识别。paraformer 走 PyTorch 后端,推理不释放 GIL,多线程并不会有实际加速,
    # 反而存在 model.generate 线程不安全的隐患,故保持串行(稳定优先)。
    if not quiet:
        print(f"    分 {len(chunks)} 段, 顺序识别...")
    results_map = {}
    for pos, (idx, chunk) in enumerate(chunks, 1):
        results_map[idx] = _recognize_chunk(model, chunk)
        if not quiet:
            print(f"\r    进度: {pos}/{len(chunks)}", end="", flush=True)

    if not quiet:
        print()  # 换行

    # 按顺序拼接结果(跟随实际 chunk 索引)
    texts = _collect_chunk_texts(chunks, results_map)

    return texts


def transcribe_funasr_diarize(model, wav_path, num_spk, quiet=False):
    """整段识别 + 说话人分人(A/B...)。整段喂入保证说话人标签全局一致。
    同一说话人的连续片段合并成完整句子,读起来更顺。"""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    target_sr = 16000
    if sr != target_sr:
        import torch
        import torchaudio.functional as F
        audio = F.resample(torch.from_numpy(audio), sr, target_sr).numpy()
    auto = num_spk <= 1  # 1=自动判断人数; >=2 = 固定 N 人
    if not quiet:
        print("    整段识别 + 自动说话人分人..." if auto else f"    整段识别 + 说话人分人({num_spk} 人)...")
    results = model.generate(input=audio, input_len=len(audio),
                             preset_spk_num=(None if auto else int(num_spk)))

    # 收集 (说话人, 文本) 序列
    segs = []
    for res in results:
        si = res.get("sentence_info", [])
        if not si:
            # 无 sentence_info(如 sensevoice 等不支持的模型),退化为普通文本
            for line in _split_sentences(res.get("text", "").strip()):
                if line:
                    segs.append((None, line))
            continue
        for s in si:
            text = (s.get("sentence") or s.get("text") or "").strip()
            if text:
                segs.append((int(s.get("spk", 0) or 0), text))

    # 合并同说话人连续片段,到句末标点或说话人切换时断行
    lines = []
    buf = ""
    cur_spk = None

    def flush():
        nonlocal buf
        if not buf:
            return
        label = chr(ord("A") + cur_spk) if cur_spk is not None and 0 <= cur_spk <= 25 else "?"
        for line in _split_sentences(buf):
            if line:
                lines.append(f"[{label}] {line}")
        buf = ""

    for spk, text in segs:
        if spk != cur_spk:
            flush()
            cur_spk = spk
        buf += text
        if text.endswith(("。", "！", "？", "…")):
            flush()
    flush()
    return lines


def transcribe_audio(model, backend, wav_path, language="zh", quiet=False, diarize=0):
    """统一转录入口。diarize>0 时做说话人分人(funasr 后端,整段识别)"""
    if backend == "whisper":
        return transcribe_whisper(model, wav_path, language)
    elif backend in ("paraformer", "sensevoice"):
        if diarize > 0:
            return transcribe_funasr_diarize(model, wav_path, int(diarize), quiet=quiet)
        return transcribe_funasr(model, wav_path, quiet=quiet)
    else:
        raise ValueError(f"未知的模型后端: {backend}")


def _output_file_for(file_path, backend):
    """根据音频文件与后端,得到转写输出的 txt 路径。
    文件名(不含扩展名)中的点号保留,避免 a.b.wav 与 a_b.wav 碰撞到同一输出;
    单扩展名文件命名与旧版一致。"""
    stem, ext = os.path.splitext(os.path.basename(file_path))
    stem = re.sub(r'[\\/:*?"<>|\r\n]', "_", stem)
    ext = ext.replace(".", "_")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    return os.path.join(output_dir, f"{stem}{ext}_{backend}_transcribed.txt")


def _scan_audio_files():
    """扫描 input/ 中的音频,返回待转写的文件列表"""
    base = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base, "input")
    files = []
    if os.path.isdir(input_dir):
        files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))
                 if os.path.splitext(f)[1].lower() in AUDIO_EXTS and not f.endswith(".16k.wav")]
    return files


def transcribe_file(file_path, model, backend, language, diarize=0, force=False):
    """转录单个音频文件。diarize>0 时做说话人分人;已有转写且未 --force 时跳过"""
    import soundfile as sf

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return

    file_path = os.path.abspath(file_path)
    output_file = _output_file_for(file_path, backend)
    if os.path.exists(output_file) and not force:
        print(f"  [跳过] 已有转写,未重复识别: {os.path.basename(output_file)}")
        return

    print(f"\n{'=' * 55}")
    print(f"  音频文件转文字")
    print(f"  文件: {os.path.basename(file_path)}")
    print(f"{'=' * 55}")

    # 对于 FunASR 后端，需要先转换为可读格式的 WAV
    tmp_path = None
    cleanup_tmp = False
    try:
        if backend in ("paraformer", "sensevoice"):
            # 尝试 soundfile 直接读取（wav/flac/ogg）
            try:
                sf.info(file_path)
                tmp_path = file_path
            except Exception:
                # soundfile 不支持（mp3/m4a 等），检查是否有已转换的 wav
                cached_wav = os.path.splitext(file_path)[0] + ".16k.wav"
                if os.path.exists(cached_wav):
                    print(f"  使用已转换的缓存: {os.path.basename(cached_wav)}")
                    tmp_path = cached_wav
                else:
                    print("  正在转换音频格式...")
                    import subprocess
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    cleanup_tmp = True
                    try:
                        result = subprocess.run(
                            ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", tmp_path],
                            capture_output=True,
                        )
                    except FileNotFoundError:
                        print(f"\n  [错误] 未找到 ffmpeg，请安装：")
                        print(f"         winget install ffmpeg")
                        print(f"         或从 https://ffmpeg.org 下载并添加到 PATH")
                        return
                    if result.returncode != 0:
                        print(f"\n  [错误] ffmpeg 转换失败:")
                        print(f"         {result.stderr.decode(errors='ignore')[-200:]}")
                        return
                    # 保存转换后的 wav 供下次使用
                    import shutil
                    shutil.copy2(tmp_path, cached_wav)
                    out_size = os.path.getsize(cached_wav)
                    print(f"  转换完成: {out_size / 1024 / 1024:.1f} MB → {os.path.basename(cached_wav)}")
            # 转录（优先走常驻服务，避免重复加载模型）
            print("  正在识别...")
            texts = transcribe_audio_dispatch(backend, tmp_path, language, model, diarize=diarize)
        else:
            # Whisper 可以直接处理大多数格式
            print("  正在识别...")
            texts = transcribe_audio_dispatch(backend, file_path, language, model, diarize=diarize)

        # 输出结果(按句拆行,便于阅读和喂给大模型)
        if texts:
            lines = []
            for t in texts:
                lines.extend(_split_sentences(t))
            print(f"\n{'-' * 55}")
            for i, line in enumerate(lines, 1):
                print(f"  {i}. {line}")

            # 保存到文件（包含扩展名和模型名，避免不同模型/格式互相覆盖）
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"=== 音频识别 - {os.path.basename(file_path)} - "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
                f.write("\n".join(lines) + "\n")

            print(f"{'-' * 55}")
            print(f"  共识别 {len(lines)} 句(已分行),结果保存在:")
            print(f"  {output_file}")
        else:
            print("\n  未识别到任何文字")

    except Exception as e:
        print(f"[错误] 转录失败: {e}")
    finally:
        if cleanup_tmp and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print(f"{'=' * 55}")


# ========== 常驻识别服务 ==========

_SERVER = None  # 模块级:当前进程连接的常驻服务客户端


class ServerClient:
    """连接常驻识别服务的 HTTP 客户端"""

    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def health(self):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    def transcribe_path(self, path, timeout=600, diarize=0):
        body = json.dumps({"path": path, "diarize": diarize}).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/transcribe_path", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "常驻服务转写失败"))
        return data.get("texts", [])

    def shutdown(self):
        try:
            req = urllib.request.Request(self.base + "/shutdown", data=b"", method="POST")
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass


def run_server(config):
    """--server 模式:加载一次模型,常驻提供转写接口"""
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port = int(config.get("server_port", DEFAULT_SERVER_PORT))
    idle_min = int(config.get("server_idle_exit_minutes", DEFAULT_SERVER_IDLE_MIN))
    backend = config.get("model_backend", "whisper")
    language = config.get("language", DEFAULT_LANGUAGE)
    diarize = int(config.get("diarize", 0) or 0)

    # 记录启动时的代码指纹,客户端据此判断服务是否需要因代码更新而重启
    code_sig = (os.path.getmtime(os.path.abspath(__file__)),
                os.path.getsize(os.path.abspath(__file__)))

    print(f"常驻服务:加载模型 {backend} ...")
    try:
        model, backend = load_model(config)
    except Exception as e:
        print(f"[错误] 常驻服务模型加载失败: {e}")
        return

    lock = threading.Lock()
    last_request = {"t": time.time()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send_json({"ok": True, "backend": backend,
                                 "language": language, "diarize": diarize,
                                 "code_sig": list(code_sig), "pid": os.getpid()})
            else:
                self._send_json({"ok": False}, 404)

        def do_POST(self):
            last_request["t"] = time.time()
            try:
                if self.path == "/transcribe_path":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    diarize = int(body.get("diarize", 0) or 0)
                    with lock:
                        texts = transcribe_audio(model, backend, body["path"], language,
                                                 quiet=True, diarize=diarize)
                    self._send_json({"ok": True, "texts": texts})
                elif self.path == "/shutdown":
                    self._send_json({"ok": True})
                    threading.Thread(target=lambda: self.server.shutdown(), daemon=True).start()
                else:
                    self._send_json({"ok": False}, 404)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    if idle_min > 0:
        def _idle_watch():
            while True:
                time.sleep(30)
                if time.time() - last_request["t"] > idle_min * 60:
                    print(f"\n常驻服务空闲超过 {idle_min} 分钟,自动退出")
                    try:
                        server.shutdown()
                    except Exception:
                        pass
                    return
        threading.Thread(target=_idle_watch, daemon=True).start()

    print(f"常驻服务就绪: 127.0.0.1:{port} (backend={backend}, pid={os.getpid()})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("常驻服务已退出")


def wait_port_free(health_fn, timeout=10.0, interval=0.3):
    """等待旧常驻服务真正释放端口(health 不可达)。返回是否已释放。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if health_fn() is None:
            return True
        time.sleep(interval)
    return health_fn() is None


def ensure_server(config):
    """确保常驻服务可用。返回 ServerClient, 失败返回 None(回退进程内模型)"""
    import sys
    port = int(config.get("server_port", DEFAULT_SERVER_PORT))
    client = ServerClient(port)
    backend = config.get("model_backend", "whisper")
    language = config.get("language", DEFAULT_LANGUAGE)
    diarize = int(config.get("diarize", 0) or 0)
    my_sig = (os.path.getmtime(os.path.abspath(__file__)),
              os.path.getsize(os.path.abspath(__file__)))

    info = client.health()
    if info and info.get("ok"):
        same_cfg = (info.get("backend") == backend and info.get("language") == language
                    and int(info.get("diarize", 0) or 0) == diarize)
        same_code = tuple(info.get("code_sig", (0, 0)) or (0, 0)) == my_sig
        if same_cfg and same_code:
            print(f"  已连接常驻识别服务 (模型: {info.get('backend')}, pid: {info.get('pid')})")
            return client
        print(f"  常驻服务需重启(配置或代码已变 {info.get('backend')} → {backend}),正在重启...")
        client.shutdown()
        # 固定 sleep 无法保证旧进程已释放端口,新进程可能 bind 失败
        if not wait_port_free(client.health, timeout=10):
            print("  [!] 旧服务未按时释放端口,新服务可能启动失败,将靠健康检查确认")
        time.sleep(0.5)

    import subprocess
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "server.log"), "a", encoding="utf-8")
    try:
        flags = subprocess.DETACHED_PROCESS if os.name == "nt" else 0
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--server"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
            creationflags=flags, close_fds=True,
        )
    except Exception as e:
        print(f"  [!] 无法后台启动常驻服务: {e},改用进程内模型")
        return None

    print("  首次运行:正在后台加载模型(约 15~30 秒)...", end="", flush=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        info = client.health()
        if info and info.get("ok") and info.get("backend") == backend \
                and int(info.get("diarize", 0) or 0) == diarize \
                and tuple(info.get("code_sig", (0, 0)) or (0, 0)) == my_sig:
            print(" 完成", flush=True)
            return client
        time.sleep(0.7)
    print("  [!] 常驻服务启动超时,改用进程内模型", flush=True)
    return None


def transcribe_audio_dispatch(backend, wav_path, language="zh", model=None, quiet=False, diarize=0):
    """优先走常驻服务,否则用本地模型转写"""
    if _SERVER is not None:
        return _SERVER.transcribe_path(wav_path, diarize=diarize)
    return transcribe_audio(model, backend, wav_path, language, quiet=quiet, diarize=diarize)


class SystemAudioCapture:
    """通过 WASAPI 环回捕获系统所有音频输出"""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.sample_rate = None
        self.channels = None
        self.chunk_size = 1024
        self.audio_buffer = []
        self.lock = threading.Lock()
        self._running = False

    def find_loopback_device(self):
        """找到默认扬声器对应的环回设备"""
        try:
            wasapi_info = self.pa.get_host_api_info_by_type(
                pyaudio.paWASAPI
            )
        except OSError:
            print("错误: 系统不支持 WASAPI")
            return None

        # 获取默认扬声器
        default_speakers = self.pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        print(f"默认扬声器: {default_speakers['name']}")

        # 精确匹配：loopback 设备名去掉 " [Loopback]" 后缀 == 默认扬声器名
        speaker_name = default_speakers["name"]
        for loopback in self.pa.get_loopback_device_info_generator():
            lb_name = loopback["name"].replace(" [Loopback]", "")
            if lb_name == speaker_name:
                print(f"环回设备: {loopback['name']}")
                return loopback

        # 备用：模糊匹配（设备名包含扬声器关键词）
        for loopback in self.pa.get_loopback_device_info_generator():
            if speaker_name.split(" (")[0] in loopback["name"]:
                print(f"环回设备(模糊): {loopback['name']}")
                return loopback

        # 最后兜底：第一个 loopback
        for loopback in self.pa.get_loopback_device_info_generator():
            print(f"环回设备(备用): {loopback['name']}")
            return loopback

        print("错误: 找不到环回音频设备")
        return None

    def start(self):
        """开始捕获音频"""
        device = self.find_loopback_device()
        if not device:
            return False

        self.sample_rate = int(device["defaultSampleRate"])
        self.channels = int(device["maxInputChannels"])

        print(f"采样率: {self.sample_rate} Hz, 声道: {self.channels}")

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            input_device_index=device["index"],
        )
        self.stream.start_stream()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("OK 开始捕获系统音频\n")
        return True

    def _read_loop(self):
        while self._running:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                with self.lock:
                    self.audio_buffer.append(data)
            except Exception:
                break

    def get_audio_chunk(self):
        """取出当前缓冲区的所有音频数据"""
        with self.lock:
            if not self.audio_buffer:
                return None
            data = b"".join(self.audio_buffer)
            self.audio_buffer.clear()
        return data

    def stop(self):
        self._running = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except Exception:
            pass
        try:
            self.pa.terminate()
        except Exception:
            pass


def save_wav(pcm_data, sample_rate, channels, filepath):
    """把 PCM 数据保存为临时 WAV 文件"""
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return filepath


def to_16k_mono(pcm_bytes, sr, channels):
    """把任意采样率/声道的 int16 PCM 转成 16kHz 单声道。
    录音只用于语音识别(模型只吃 16k),降下来文件可缩小约 6 倍,且转写时无需重采样。"""
    if not pcm_bytes:
        return b""
    x = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)  # 立体声→单声道
    if sr == 16000:
        return x.astype(np.int16).tobytes()
    # 线性插值重采样到 16k
    n_out = max(1, round(len(x) * 16000 / sr))
    xf = x.astype(np.float32)
    idx = np.linspace(0, len(x) - 1, n_out)
    i0 = idx.astype(np.int64)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = (idx - i0).astype(np.float32)
    y = xf[i0] * (1 - frac) + xf[i1] * frac
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


def _split_sentences(text, max_len=150):
    """把长文本按句子标点拆成短行(每句一行),单句超长再按逗号就近切分。
    便于在编辑器里阅读、也适合直接喂给大语言模型(意思完整保留)。"""
    lines = []
    for part in re.split(r'(?<=[。！？…])', text):
        part = part.strip()
        if not part:
            continue
        while len(part) > max_len:
            head = part[:max_len]
            cut = max(head.rfind('，'), head.rfind('。'), head.rfind('！'),
                      head.rfind('；'), head.rfind('、'), head.rfind(','),
                      head.rfind(';'), head.rfind('？'))
            if cut <= 0:
                cut = max_len
            else:
                cut += 1  # 保留标点
            lines.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            lines.append(part)
    return lines


def check_silence(pcm_data, threshold=50):
    """检测是否静音（避免识别无声段产生乱码）"""
    audio = np.frombuffer(pcm_data, dtype=np.int16)
    if len(audio) == 0:
        return True
    return np.abs(audio).mean() < threshold


def _save_transcribed(wav_path, texts, backend):
    """把转写结果保存为 txt(每句一行),返回输出文件路径。
    统一走 output/ 目录(与文件模式 _output_file_for 一致),避免录制转写后又被批量模式重复转写。"""
    if not texts:
        print("\n  未识别到文字")
        return None
    lines = []
    for t in texts:
        lines.extend(_split_sentences(t))
    out_file = _output_file_for(wav_path, backend)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"=== 录音转写 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\n{'=' * 55}")
    print(f"转写完成({len(lines)} 句,已按句分行),文案保存在:")
    print(f"  {out_file}")
    print(f"{'=' * 55}")
    print("预览:")
    for t in lines[:10]:
        print(f"  {t}")
    if len(lines) > 10:
        print(f"  ... 共 {len(lines)} 句")
    return out_file


def _transcribe_recorded(config, wav_path):
    """对已录制的音频做离线整段转写(断句级,最准)。优先常驻服务,失败回退进程内模型"""
    import soundfile as sf
    backend = config.get("model_backend", "whisper")
    language = config.get("language", DEFAULT_LANGUAGE)
    diarize = int(config.get("diarize", 0) or 0)

    try:
        dur = sf.info(wav_path).duration
    except Exception:
        dur = 0
    extra = f", {diarize} 人分人" if diarize > 0 else ""
    print(f"\n  整段转写中({dur/60:.1f} 分钟音频,自动断句加标点{extra},约需几十秒~2分钟)...", flush=True)

    global _SERVER
    if _SERVER is None:
        _SERVER = ensure_server(config)

    # 优先走常驻服务(快);失败则回退进程内(稳)
    if _SERVER is not None:
        try:
            texts = _SERVER.transcribe_path(wav_path, timeout=600, diarize=diarize)
            return _save_transcribed(wav_path, texts, backend)
        except Exception as e:
            print(f"  [!] 常驻服务转写失败({e}),改用进程内模型...", flush=True)
            _SERVER = None

    try:
        model, backend = load_model(config)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}", flush=True)
        print("  录音文件仍保留,可稍后用 run.bat --file 重新转写。", flush=True)
        return
    try:
        texts = transcribe_audio(model, backend, wav_path, language, quiet=True, diarize=diarize)
    except Exception as e:
        if diarize > 0:
            print(f"  [!] 分人转写出错({e}),改为不带分人的普通转写...", flush=True)
            try:
                texts = transcribe_audio(model, backend, wav_path, language, quiet=True, diarize=0)
            except Exception as e2:
                print(f"[错误] 转写失败: {e2}", flush=True)
                print("  录音文件仍保留,可稍后用 run.bat --file 重新转写。", flush=True)
                return
        else:
            print(f"[错误] 转写失败: {e}", flush=True)
            print("  录音文件仍保留,可稍后用 run.bat --file 重新转写。", flush=True)
            return
    _save_transcribed(wav_path, texts, backend)


def _record_one_session(config, record_dir, idle_timeout):
    """录制单个视频到 record_dir,返回 wav 路径(启动失败返回 None)"""
    REC_SR = 16000  # 录音统一转成 16kHz 单声道(模型输入格式,文件小 6 倍)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = os.path.join(record_dir, f"record_{start_time}.wav")

    print("=" * 55)
    print(f"  开始录音(16kHz 单声道) → {os.path.basename(record_file)}")
    if idle_timeout > 0:
        print(f"  无声音自动停止: {idle_timeout // 60} 分钟")
    else:
        print("  无声音自动停止: 已禁用")
    print("  播放视频/音频,完成后按任意键停止")
    print("=" * 55)

    capture = SystemAudioCapture()
    if not capture.start():
        return None

    last_audio_time = time.time()
    wf = None
    try:
        wf = wave.open(record_file, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(REC_SR)
    except Exception as e:
        print(f"[错误] 无法创建录音文件: {e}")
        capture.stop()
        return None

    frames = 0
    try:
        import msvcrt
        while True:
            pcm = capture.get_audio_chunk()
            if pcm is not None and len(pcm) > 0:
                pcm16 = to_16k_mono(pcm, capture.sample_rate, capture.channels)
                if pcm16:
                    wf.writeframes(pcm16)
                    frames += len(pcm16)
                if not check_silence(pcm):
                    last_audio_time = time.time()
            dur = frames / (REC_SR * 2)
            print(f"\r  已录 {dur/60:.1f} 分钟(按任意键停止)", end="", flush=True)
            # 按任意键优雅停止(避免 Ctrl+C 把批处理窗口一起杀掉)
            try:
                if msvcrt.kbhit():
                    try:
                        msvcrt.getch()
                    except Exception:
                        pass
                    print()
                    break
            except Exception:
                pass  # 非控制台环境忽略,靠空闲超时或 Ctrl+C
            time.sleep(0.5)
            if idle_timeout > 0 and time.time() - last_audio_time >= idle_timeout:
                print(f"\n  ⏸  已 {idle_timeout // 60} 分钟无声音,自动停止")
                break
    except KeyboardInterrupt:
        print()
    finally:
        capture.stop()
        try:
            wf.close()
        except Exception:
            pass

    dur_sec = os.path.getsize(record_file) / (REC_SR * 2)
    print(f"\n  录音完成:{dur_sec/60:.1f} 分钟 → {record_file}")
    return record_file


def record_mode(config):
    """纯录制模式:循环常驻,一个接一个录音到 input/,每个视频一个文件,窗口一直开着。
    录完一个后可继续录下一个;按 T 立即转写当前;按 Q 退出(稍后用 run.bat 批量转写)。"""
    idle_timeout = config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT)
    record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    os.makedirs(record_dir, exist_ok=True)

    print("=" * 55)
    print("  纯录制模式:循环录音,窗口常驻")
    print("  录音存入 input/,每个视频一个独立文件,按 Q 退出")
    print("=" * 55)

    while True:
        record_file = _record_one_session(config, record_dir, idle_timeout)
        if record_file is None:
            break
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()  # 清空按键缓冲
            print("\n  按 任意键 录下一个  |  按 T 转写当前  |  按 Q 退出")
            key = msvcrt.getch().decode("utf-8", "ignore").lower()
        except Exception:
            key = "q"
        if key == "q":
            break
        if key == "t":
            try:
                _transcribe_recorded(config, record_file)
            except BaseException as e:
                print(f"[错误] 转写失败: {e}", flush=True)

    print("\n  录制完成。之后用 run.bat 选批量转写,或 run.bat --file 指定文件(已转的会自动跳过)。")


def main():
    import sys

    # 常驻服务专用子命令
    if "--stop-server" in sys.argv:
        _cfg = load_config()
        ServerClient(int(_cfg.get("server_port", DEFAULT_SERVER_PORT))).shutdown()
        print("已发送停止信号给常驻识别服务")
        return
    if "--server" in sys.argv:
        _cfg = load_config()
        if not _cfg or "model_backend" not in _cfg:
            _cfg = prompt_model_selection()
        run_server(_cfg)
        return

    # 检查 --file 参数
    file_paths = []
    INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")

    if "--file" in sys.argv:
        # 显式指定文件,直接进入文件模式
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
            path = sys.argv[idx + 1]
            if os.path.isabs(path) or os.sep in path or "/" in path:
                file_paths = [path]
            else:
                input_path = os.path.join(INPUT_DIR, path)
                file_paths = [input_path] if os.path.exists(input_path) else [path]
        else:
            # 无参数:批量模式,扫描 input/ 中的音频
            print("  批量模式:扫描 input/ 中的音频")
            file_paths = _scan_audio_files()
    elif "--live" in sys.argv:
        # 强制实时模式，跳过文件检测
        pass
    elif "--record" in sys.argv:
        # 纯录制模式,在配置加载后处理
        pass
    else:
        # 扫描 input/ 文件夹
        input_files = []
        if os.path.isdir(INPUT_DIR):
            input_files = sorted(
                f for f in os.listdir(INPUT_DIR)
                if os.path.splitext(f)[1].lower() in AUDIO_EXTS
                and not f.endswith(".16k.wav")
            )

        if input_files:
            # 有文件，让用户选择模式
            print("=" * 55)
            print(f"  input/ 文件夹中有 {len(input_files)} 个音频文件：")
            print("=" * 55)
            for f in input_files:
                print(f"    {f}")
            print()
            print("  [1] 转写这些文件(批量)")
            print("  [2] 录音+自动转写(先录,停止后整段转写)")
            print("  [3] 实时字幕(边看边出字)")
            while True:
                mode = input("\n请选择模式 (1/2/3) [1]: ").strip() or "1"
                if mode in ("1", "2", "3"):
                    break
                print("无效输入，请重新选择")
            if mode == "1":
                file_paths = [os.path.join(INPUT_DIR, f) for f in input_files]
            elif mode == "2":
                sys.argv.append("--record")
        else:
            # 无文件:选择纯录制或实时
            print("=" * 55)
            print("  选择模式：")
            print("  [1] 录音+自动转写(先录,停止后整段转写) — 推荐")
            print("  [2] 实时字幕(边看边出字)")
            print("  [3] 批量转写(重新扫描 input/)")
            while True:
                mode = input("\n请选择模式 (1/2/3) [1]: ").strip() or "1"
                if mode in ("1", "2", "3"):
                    break
                print("无效输入，请重新选择")
            if mode == "1":
                sys.argv.append("--record")
            elif mode == "3":
                print("  批量模式:扫描 input/ 中的音频")
                file_paths = _scan_audio_files()
                if not file_paths:
                    print("  input/ 中没有可转写的音频文件。")
                    return

    # 命令行显式 --file 批量但扫不到待转文件时明确退出,避免静默掉进实时模式
    if "--file" in sys.argv and not file_paths:
        print("  input/ 中没有可转写的音频文件。")
        return

    # 1. 加载配置，已有配置直接使用，首次才弹出选择
    config = load_config()
    if "model_backend" not in config or "language" not in config:
        config = prompt_model_selection()
    else:
        backend_name = {v["backend"]: v["name"] for v in MODEL_OPTIONS.values()}.get(config["model_backend"], config["model_backend"])
        lang_name = {"zh": "中文", "en": "English"}.get(config.get("language", "zh"), config.get("language", "zh"))
        print(f"  使用配置: {backend_name} | {lang_name}")

    backend = config["model_backend"]
    language = config.get("language", DEFAULT_LANGUAGE)

    # 纯录制模式:先录音,停止后再离线转写(此处不加载模型,点开即录)
    if "--record" in sys.argv:
        record_mode(config)
        return

    # 文件模式:过滤已转写的文件(跳过已有文字),全部完成则无需加载模型
    if file_paths:
        force = "--force" in sys.argv
        file_paths = [p for p in file_paths
                      if force or not os.path.exists(_output_file_for(p, config.get("model_backend", "whisper")))]
        if not file_paths:
            print("  全部音频已有转写,跳过(加 --force 可强制重转)。")
            return

    # 模型：优先连接常驻服务(秒连),否则进程内加载
    global _SERVER
    _SERVER = ensure_server(config)
    model = None
    if _SERVER is None:
        try:
            model, backend = load_model(config)
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            return

    # --- 文件转录模式 ---
    if file_paths:
        backend_name = {v["backend"]: v["name"] for v in MODEL_OPTIONS.values()}.get(backend, backend)
        lang_name = {"zh": "中文", "en": "English"}.get(language, language)
        print("=" * 55)
        print(f"  音频文件转文字 | {backend_name} | {lang_name}")
        print(f"  共 {len(file_paths)} 个待转文件")
        print("=" * 55)
        force = "--force" in sys.argv
        for path in file_paths:
            transcribe_file(path, model, backend, language,
                            diarize=int(config.get("diarize", 0) or 0), force=force)
        return

    # --- 实时系统音频模式 ---
    backend_name = {v["backend"]: v["name"] for v in MODEL_OPTIONS.values()}.get(backend, backend)
    chunk_seconds = config.get("chunk_seconds", DEFAULT_CHUNK_SECONDS)
    chunk_desc = CHUNK_HINT.get(chunk_seconds, "自定义间隔")
    idle_timeout = config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT)
    lang_name = {"zh": "中文", "en": "English"}.get(language, language)

    print("=" * 55)
    print("  系统音频实时语音识别 → TXT")
    print(f"  模型: {backend_name} | 语言: {lang_name} | 设备: {DEVICE}")
    print(f"  识别间隔: {chunk_seconds}s — {chunk_desc}")
    if idle_timeout > 0:
        print(f"  无声音退出: {idle_timeout // 60} 分钟")
    else:
        print(f"  无声音退出: 已禁用")
    print("=" * 55)

    # 2. 启动音频捕获
    capture = SystemAudioCapture()
    if not capture.start():
        return

    # 4. 初始化输出文件
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"captions_{start_time}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== 音频识别记录 - "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"===\n\n")

    print(f"每 {chunk_seconds} 秒提交一个识别窗口(内部自动断句加标点),按 Ctrl+C 停止")
    print(f"提示: 可修改 {CONFIG_FILE} 中的 chunk_seconds / idle_timeout / language")
    print(f"      3s=低延迟  5s=均衡(推荐)  idle_timeout=0 禁用  language=zh/en")
    print(f"输出文件: {output_file}\n")
    print("-" * 55)

    all_text = []
    last_text = ""
    last_audio_time = time.time()  # 最后一次检测到声音的时间

    # 全程录音(供结束时整段重识别,统一 16kHz 单声道减小体积)
    SESS_SR = 16000
    session_wav = os.path.join(output_dir, f"session_{start_time}.wav")
    session_wf = None
    try:
        session_wf = wave.open(session_wav, "wb")
        session_wf.setnchannels(1)
        session_wf.setsampwidth(2)
        session_wf.setframerate(SESS_SR)
    except Exception:
        session_wf = None

    pending = bytearray()  # 待识别窗口(累积到 chunk_seconds 才提交,形成完整句子)
    flush_bytes = max(2, chunk_seconds) * capture.sample_rate * capture.channels * 2

    try:
        while True:
            t0 = time.time()

            # 取出音频数据
            pcm_data = capture.get_audio_chunk()
            if pcm_data is not None and len(pcm_data) > 0:
                # 全程录音
                if session_wf is not None:
                    try:
                        session_wf.writeframes(
                            to_16k_mono(pcm_data, capture.sample_rate, capture.channels))
                    except Exception:
                        pass
                pending += pcm_data

            # 窗口累积到 chunk_seconds 后提交识别
            if len(pending) >= flush_bytes:
                seg = bytes(pending)
                pending.clear()
                # 跳过静音段
                if not check_silence(seg):
                    last_audio_time = time.time()  # 有声音，刷新计时
                    tmp_path = None
                    try:
                        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        tmp_path = tmp.name
                        tmp.close()
                        save_wav(seg, capture.sample_rate, capture.channels, tmp_path)

                        # 统一转录(静默模式)
                        texts = transcribe_audio_dispatch(backend, tmp_path, language, model, quiet=True, diarize=0)

                        # 输出结果
                        for text in texts:
                            if not text or text == last_text:
                                continue
                            last_text = text
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] {text}")

                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(text + "\n")
                            all_text.append(text)

                    except Exception as e:
                        print(f"[!] 识别出错: {e}")
                        # 常驻服务可能已中途退出(空闲退出/被手动停止/崩溃),
                        # 回退为进程内模型,避免整个会话后续窗口全部失败
                        if _SERVER is not None:
                            print("  常驻服务不可用,切换为进程内模型...")
                            _SERVER = None
                            try:
                                model, backend = load_model(config)
                            except Exception as e2:
                                print(f"[错误] 进程内模型加载失败: {e2}")
                                model = None
                                break
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)

            # 快速轮询,维持窗口按 chunk_seconds 累积
            elapsed = time.time() - t0
            time.sleep(max(0.2, 0.5 - elapsed))

            # 检查空闲超时
            if idle_timeout > 0:
                idle_seconds = time.time() - last_audio_time
                if idle_seconds >= idle_timeout:
                    idle_min = int(idle_seconds // 60)
                    print(f"\n  ⏸  已 {idle_min} 分钟无声音，自动停止")
                    break

    except KeyboardInterrupt:
        print()
    finally:
        capture.stop()
        if session_wf is not None:
            try:
                session_wf.close()
            except Exception:
                pass

    print(f"{'=' * 55}")
    print(f"已停止！共识别 {len(all_text)} 条")
    print(f"实时结果保存在: {output_file}")
    print(f"{'=' * 55}")

    # 结束时整段重识别(断句级,最准文案)
    if session_wf is not None and os.path.exists(session_wav) and os.path.getsize(session_wav) > 44:
        try:
            ans = input("\n是否整段重识别以获得最准确文案? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans in ("", "y", "yes"):
            dur_sec = os.path.getsize(session_wav) / (SESS_SR * 2)
            print(f"\n  整段重识别中({dur_sec/60:.1f} 分钟音频,自动断句加标点,约需几秒~1分钟)...")
            try:
                final_texts = transcribe_audio_dispatch(backend, session_wav, language, model, quiet=True,
                                                        diarize=int(config.get("diarize", 0) or 0))
                if final_texts:
                    lines = []
                    for t in final_texts:
                        lines.extend(_split_sentences(t))
                    final_file = os.path.join(output_dir, f"captions_{start_time}_final.txt")
                    with open(final_file, "w", encoding="utf-8") as f:
                        f.write(f"=== 整段重识别 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
                        f.write("\n".join(lines) + "\n")
                    print(f"\n{'=' * 55}")
                    print(f"整段识别完成({len(lines)} 句,已分行),最准文案保存在:")
                    print(f"  {final_file}")
                    print(f"{'=' * 55}")
                    print("预览:")
                    for t in lines[:10]:
                        print(f"  {t}")
                    if len(lines) > 10:
                        print(f"  ... 共 {len(lines)} 句")
                else:
                    print("\n  整段重识别未识别到文字")
            except Exception as e:
                print(f"[!] 整段重识别失败: {e}")


if __name__ == "__main__":
    main()
