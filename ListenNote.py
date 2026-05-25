import re
import numpy as np
import threading
import time
import wave
import tempfile
import os
import json
import logging
from datetime import datetime

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MODELSCOPE_DISABLE_PROGRESS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import pyaudiowpatch as pyaudio

# ========== 默认配置 ==========
DEFAULT_CHUNK_SECONDS = 5       # 默认识别间隔：3s=低延迟 5s=均衡 10s=高准确率
DEFAULT_IDLE_TIMEOUT = 300      # 无声音自动退出秒数，0=禁用
DEFAULT_LANGUAGE = "zh"         # 默认识别语言：zh=中文 en=英文
DEVICE = "cpu"                  # cpu / cuda(N卡加速)
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# 间隔选择提示
CHUNK_HINT = {
    3: "低延迟模式，适合实时对话",
    5: "均衡模式（推荐），兼顾实时性与准确率",
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
        # 去掉行内注释
        in_string = False
        for i, ch in enumerate(line):
            if ch == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
            if not in_string and line[i:i+2] in ("//",):
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
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        config = json.loads(_strip_json_comments(text))
        # 自动补全缺失字段并写回
        updated = False
        for key, val in defaults.items():
            if key not in config:
                config[key] = val
                updated = True
        if updated:
            save_config(config)
        return config
    return {}


def save_config(config):
    """保存配置到 config.json"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


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

    config = {
        "model_backend": selected["backend"],
        "language": language,
        "chunk_seconds": DEFAULT_CHUNK_SECONDS,
        "idle_timeout": DEFAULT_IDLE_TIMEOUT,
    }

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
        sys.stderr.close()
        sys.stderr = _stderr
        print(f"[错误] 未安装 funasr，请运行：")
        print(f"  pip install funasr modelscope torch torchaudio onnxruntime")
        raise
    finally:
        sys.stderr.close()
        sys.stderr = _stderr

    print(f"OK 模型加载完成\n")
    return model


def load_model(config):
    """根据配置加载对应模型，返回 (model, backend)"""
    backend = config.get("model_backend", "whisper")

    if backend == "whisper":
        size = config.get("whisper_size", "small")
        model = load_whisper_model(size)
        return model, "whisper"

    elif backend == "paraformer":
        model = load_funasr_model(
            "paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
        )
        return model, "paraformer"

    elif backend == "sensevoice":
        model = load_funasr_model(
            "iic/SenseVoiceSmall",
            trust_remote_code=True,
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
            text = re.sub(r'^(<\|[^|]+\|>)+', '', text).strip()
            if text:
                texts.append(text)
    return texts


def transcribe_funasr(model, wav_path):
    """FunASR 转录（Paraformer / SenseVoice），多线程并行识别"""
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 显示音频信息
    try:
        info = sf.info(wav_path)
        duration = info.duration
        sr = info.samplerate
        print(f"    音频时长: {duration/60:.1f} 分钟, 采样率: {sr} Hz")
    except Exception as e:
        print(f"    [错误] 无法读取音频文件: {e}")
        return []

    # 读取所有分段（内存可控：每分钟 16kHz float32 ≈ 3.8MB）
    chunk_seconds = 60
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
        chunks.append((i, chunk))

    # 并行识别（ONNX 推理释放 GIL，线程可真正并行）
    workers = min(len(chunks), 4)
    print(f"    分 {len(chunks)} 段, {workers} 线程并行识别...")
    results_map = {}
    done_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_recognize_chunk, model, chunk): idx
                   for idx, chunk in chunks}
        for future in as_completed(futures):
            idx = futures[future]
            results_map[idx] = future.result()
            done_count += 1
            print(f"\r    进度: {done_count}/{len(chunks)}", end="", flush=True)

    print()  # 换行

    # 按顺序拼接结果
    texts = []
    for i in range(len(chunks)):
        texts.extend(results_map.get(i, []))

    return texts


def transcribe_audio(model, backend, wav_path, language="zh"):
    """统一转录入口"""
    if backend == "whisper":
        return transcribe_whisper(model, wav_path, language)
    elif backend in ("paraformer", "sensevoice"):
        return transcribe_funasr(model, wav_path)
    else:
        raise ValueError(f"未知的模型后端: {backend}")


def transcribe_file(file_path, model, backend, language):
    """转录单个音频文件"""
    import soundfile as sf

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return

    file_path = os.path.abspath(file_path)
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
            # 转录
            print("  正在识别...")
            texts = transcribe_audio(model, backend, tmp_path, language)
        else:
            # Whisper 可以直接处理大多数格式
            print("  正在识别...")
            texts = transcribe_audio(model, backend, file_path, language)

        # 输出结果
        if texts:
            print(f"\n{'-' * 55}")
            full_text = ""
            for i, text in enumerate(texts, 1):
                print(f"  {i}. {text}")
                full_text += text + "\n"

            # 保存到文件（包含扩展名和模型名，避免不同模型/格式互相覆盖）
            fname = os.path.basename(file_path).replace(".", "_")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{fname}_{backend}_transcribed.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"=== 音频识别 - {os.path.basename(file_path)} - "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
                f.write(full_text)

            print(f"{'-' * 55}")
            print(f"  共识别 {len(texts)} 段，结果保存在:")
            print(f"  {output_file}")
        else:
            print("\n  未识别到任何文字")

    except Exception as e:
        print(f"[错误] 转录失败: {e}")
    finally:
        if cleanup_tmp and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print(f"{'=' * 55}")

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


def check_silence(pcm_data, threshold=50):
    """检测是否静音（避免识别无声段产生乱码）"""
    audio = np.frombuffer(pcm_data, dtype=np.int16)
    if len(audio) == 0:
        return True
    return np.abs(audio).mean() < threshold


def main():
    import sys

    # 检查 --file 参数
    file_paths = []
    INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".wma", ".opus"}

    if "--file" in sys.argv:
        # 显式指定文件，直接进入文件模式
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            path = sys.argv[idx + 1]
            if os.path.isabs(path) or os.sep in path or "/" in path:
                file_paths = [path]
            else:
                input_path = os.path.join(INPUT_DIR, path)
                file_paths = [input_path] if os.path.exists(input_path) else [path]
        else:
            print("[错误] --file 后需要指定文件路径")
            print("用法: python ListenNote.py --file <音频文件>")
            print(f"  文件可放在 {INPUT_DIR} 目录中，直接用文件名引用")
            return
    elif "--live" in sys.argv:
        # 强制实时模式，跳过文件检测
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
            print("  [1] 转录这些文件")
            print("  [2] 实时系统音频识别")
            while True:
                mode = input("\n请选择模式 (1/2) [1]: ").strip() or "1"
                if mode in ("1", "2"):
                    break
                print("无效输入，请重新选择")
            if mode == "1":
                file_paths = [os.path.join(INPUT_DIR, f) for f in input_files]

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

    # --- 文件转录模式 ---
    if file_paths:
        backend_name = {v["backend"]: v["name"] for v in MODEL_OPTIONS.values()}.get(backend, backend)
        lang_name = {"zh": "中文", "en": "English"}.get(language, language)
        print("=" * 55)
        print(f"  音频文件转文字 | {backend_name} | {lang_name}")
        print(f"  共 {len(file_paths)} 个文件")
        print("=" * 55)
        try:
            model, backend = load_model(config)
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            return
        for path in file_paths:
            transcribe_file(path, model, backend, language)
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

    # 2. 加载模型
    try:
        model, backend = load_model(config)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return

    # 3. 启动音频捕获
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

    print(f"每 {chunk_seconds} 秒识别一次，按 Ctrl+C 停止")
    print(f"提示: 可修改 {CONFIG_FILE} 中的 chunk_seconds / idle_timeout / language")
    print(f"      3s=低延迟  5s=均衡(推荐)  idle_timeout=0 禁用  language=zh/en")
    print(f"输出文件: {output_file}\n")
    print("-" * 55)

    all_text = []
    last_text = ""
    last_audio_time = time.time()  # 最后一次检测到声音的时间

    try:
        while True:
            t0 = time.time()

            # 取出音频数据
            pcm_data = capture.get_audio_chunk()
            if pcm_data is not None and len(pcm_data) >= 1000:
                # 跳过静音段
                if not check_silence(pcm_data):
                    last_audio_time = time.time()  # 有声音，刷新计时
                    # 保存临时 WAV 并识别
                    tmp_path = None
                    try:
                        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        tmp_path = tmp.name
                        tmp.close()
                        save_wav(pcm_data, capture.sample_rate, capture.channels, tmp_path)

                        # 统一转录
                        texts = transcribe_audio(model, backend, tmp_path, language)

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
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)

            # 扣除处理耗时，维持稳定间隔
            elapsed = time.time() - t0
            sleep_time = max(0.5, chunk_seconds - elapsed)
            time.sleep(sleep_time)

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

    print(f"{'=' * 55}")
    print(f"已停止！共识别 {len(all_text)} 条")
    print(f"结果保存在: {output_file}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
