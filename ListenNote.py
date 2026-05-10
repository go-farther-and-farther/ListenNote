import numpy as np
import threading
import time
import wave
import tempfile
import os
from datetime import datetime
from faster_whisper import WhisperModel

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

import pyaudiowpatch as pyaudio

# ========== 配置 ==========
OUTPUT_FILE = "captions.txt"
CHUNK_SECONDS = 15          # 每几秒识别一次（越短越实时，越长越准）
MODEL_SIZE = "medium"       # tiny/base/small/medium/large-v3
                            # medium 中文效果好，tiny 最快
DEVICE = "cpu"              # cpu / cuda(N卡加速)
LANGUAGE = "zh"             # 中文
# ===========================

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

        # 直接用默认输出设备的索引找 loopback
        for loopback in self.pa.get_loopback_device_info_generator():
            if (loopback["name"].startswith(
                    default_speakers["name"].split(" (")[0])):
                print(f"环回设备: {loopback['name']}")
                return loopback

        # 备用：尝试第一个 loopback
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


def check_silence(pcm_data, threshold=500):
    """检测是否静音（避免识别无声段产生乱码）"""
    audio = np.frombuffer(pcm_data, dtype=np.int16)
    if len(audio) == 0:
        return True
    return np.abs(audio).mean() < threshold


def main():
    print("=" * 55)
    print("  系统音频实时语音识别 → TXT")
    print(f"  模型: {MODEL_SIZE} | 设备: {DEVICE}")
    print("=" * 55)
    print("正在加载 Whisper 模型...")
    print("(首次运行需要下载模型，请等待进度条完成)\n")

    # 加载模型
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type="int8" if DEVICE == "cpu" else "float16",
    )
    print("OK 模型加载完成\n")

    # 启动音频捕获
    capture = SystemAudioCapture()
    if not capture.start():
        return

    # 初始化输出文件（带时间戳，每次运行独立文件）
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"captions_{start_time}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== 音频识别记录 - "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"===\n\n")

    print(f"每 {CHUNK_SECONDS} 秒识别一次，按 Ctrl+C 停止\n")
    print(f"输出文件: {output_file}\n")
    print("-" * 55)

    all_text = []  # 存储所有识别文本用于拼接

    try:
        while True:
            time.sleep(CHUNK_SECONDS)

            # 1. 取出音频数据
            pcm_data = capture.get_audio_chunk()
            if pcm_data is None or len(pcm_data) < 1000:
                continue

            # 2. 跳过静音段
            if check_silence(pcm_data):
                continue

            # 3. 保存临时 WAV 并识别
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                )
                tmp_path = tmp.name
                tmp.close()
                save_wav(pcm_data, capture.sample_rate,
                         capture.channels, tmp_path)

                # 4. Whisper 识别
                segments, info = model.transcribe(
                    tmp_path,
                    language=LANGUAGE,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300
                    ),
                )

                # 5. 输出结果
                for segment in segments:
                    text = segment.text.strip()
                    if not text:
                        continue

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

    except KeyboardInterrupt:
        print(f"\n{'=' * 55}")
        print(f"已停止！共识别 {len(all_text)} 条")
        print(f"结果保存在: {output_file}")
        print(f"{'=' * 55}")

    finally:
        capture.stop()


if __name__ == "__main__":
    main()
