"""ListenNote.py 单元测试（纯逻辑部分，不需要模型/网络/音频设备）

覆盖审阅发现的 5 个问题的修复：
- B1 _collect_chunk_texts   按实际 chunk 索引收集结果
- B2 _output_file_for       输出文件名不碰撞
- B3 _funasr_extra_kwargs   仅 paraformer 加载声纹模型
- B4 wait_port_free         等旧服务真正释放端口
- B5 merge_selection_config 重选模型保留用户自定义配置
"""
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ListenNote as ln


class TestOutputFileFor(unittest.TestCase):
    """B2: 不同输入文件不得碰撞到同一个输出文件"""

    def test_dot_vs_underscore_names_do_not_collide(self):
        a = ln._output_file_for(os.path.join("input", "a.b.wav"), "paraformer")
        b = ln._output_file_for(os.path.join("input", "a_b.wav"), "paraformer")
        self.assertNotEqual(a, b)

    def test_single_extension_name_keeps_old_format(self):
        p = ln._output_file_for(os.path.join("input", "meeting.mp3"), "paraformer")
        self.assertTrue(p.endswith("meeting_mp3_paraformer_transcribed.txt"))
        self.assertTrue(os.path.basename(os.path.dirname(p)) == "output")

    def test_same_file_same_name(self):
        a = ln._output_file_for(os.path.join("input", "x.y.wav"), "sensevoice")
        b = ln._output_file_for(os.path.join("input", "x.y.wav"), "sensevoice")
        self.assertEqual(a, b)


class TestCollectChunkTexts(unittest.TestCase):
    """B1: 结果收集必须跟随实际 chunk 索引，跳段不得丢文本"""

    def test_sequential_chunks(self):
        chunks = [(0, None), (1, None), (2, None)]
        rmap = {0: ["a"], 1: ["b"], 2: ["c"]}
        self.assertEqual(ln._collect_chunk_texts(chunks, rmap), ["a", "b", "c"])

    def test_middle_chunk_skipped_keeps_later_text(self):
        # 旧逻辑 for i in range(len(chunks)) 会丢掉 chunk 2 的结果
        chunks = [(0, None), (2, None)]
        rmap = {0: ["a"], 2: ["c"]}
        self.assertEqual(ln._collect_chunk_texts(chunks, rmap), ["a", "c"])

    def test_empty(self):
        self.assertEqual(ln._collect_chunk_texts([], {}), [])


class TestFunasrExtraKwargs(unittest.TestCase):
    """B3: 只有 paraformer 传声纹模型，其他后端不得白加载"""

    def test_paraformer_with_diarize_includes_spk_model(self):
        kw = ln._funasr_extra_kwargs("paraformer", 1)
        self.assertIn("spk_model", kw)

    def test_sensevoice_with_diarize_excludes_spk_model(self):
        kw = ln._funasr_extra_kwargs("sensevoice", 1)
        self.assertNotIn("spk_model", kw)

    def test_diarize_off_no_kwargs(self):
        self.assertEqual(ln._funasr_extra_kwargs("paraformer", 0), {})


class TestWaitPortFree(unittest.TestCase):
    """B4: 必须等到旧服务真正不可达才返回"""

    def test_returns_true_once_health_goes_down(self):
        calls = {"n": 0}

        def health():
            calls["n"] += 1
            return {"ok": True} if calls["n"] < 3 else None

        self.assertTrue(ln.wait_port_free(health, timeout=5, interval=0.01))

    def test_returns_false_on_timeout(self):
        def health():
            return {"ok": True}

        t0 = time.time()
        self.assertFalse(ln.wait_port_free(health, timeout=0.3, interval=0.05))
        self.assertGreaterEqual(time.time() - t0, 0.3)


class TestMergeSelectionConfig(unittest.TestCase):
    """B5: 重选模型必须保留用户自定义配置"""

    def test_preserves_user_customized_fields(self):
        existing = {
            "model_backend": "whisper", "language": "en",
            "chunk_seconds": 10, "idle_timeout": 600,
            "diarize": 2, "server_port": 18000,
            "server_idle_exit_minutes": 30, "whisper_size": "small",
        }
        cfg = ln.merge_selection_config(existing, "paraformer", "zh")
        self.assertEqual(cfg["model_backend"], "paraformer")
        self.assertEqual(cfg["language"], "zh")
        self.assertEqual(cfg["chunk_seconds"], 10)
        self.assertEqual(cfg["idle_timeout"], 600)
        self.assertEqual(cfg["diarize"], 2)
        self.assertEqual(cfg["server_port"], 18000)
        self.assertEqual(cfg["server_idle_exit_minutes"], 30)

    def test_fills_defaults_for_missing_fields(self):
        cfg = ln.merge_selection_config({}, "paraformer", "zh")
        self.assertEqual(cfg["language"], "zh")
        self.assertEqual(cfg["chunk_seconds"], ln.DEFAULT_CHUNK_SECONDS)
        self.assertEqual(cfg["idle_timeout"], ln.DEFAULT_IDLE_TIMEOUT)
        self.assertEqual(cfg["diarize"], ln.DEFAULT_DIARIZE)
        self.assertEqual(cfg["server_port"], ln.DEFAULT_SERVER_PORT)
        self.assertEqual(cfg["server_idle_exit_minutes"], ln.DEFAULT_SERVER_IDLE_MIN)

    def test_whisper_size_updated_when_given(self):
        cfg = ln.merge_selection_config({"model_backend": "paraformer"},
                                        "whisper", "en", whisper_size="base")
        self.assertEqual(cfg["whisper_size"], "base")


class TestLocalModelDir(unittest.TestCase):
    """离线优先:已完整缓存的模型必须解析为本地路径,绕开联网解析"""

    def setUp(self):
        import shutil
        import tempfile
        # 不用 TemporaryDirectory:其清理会 chmod,在受限环境下报拒绝访问
        base = os.path.join(tempfile.gettempdir(), "lntest_hub")
        shutil.rmtree(base, ignore_errors=True)
        os.makedirs(base)
        self.addCleanup(shutil.rmtree, base, True)
        self._tmp_name = base
        self._old_hub = ln.MODELSCOPE_HUB_DIR
        ln.MODELSCOPE_HUB_DIR = base
        self.addCleanup(setattr, ln, "MODELSCOPE_HUB_DIR", self._old_hub)

    def _make_cached(self, model_id, with_config=True):
        d = os.path.join(self._tmp_name, *model_id.split("/"))
        os.makedirs(d, exist_ok=True)
        if with_config:
            with open(os.path.join(d, "configuration.json"), "w") as f:
                f.write("{}")
        return d

    def test_complete_cache_returns_local_path(self):
        mid = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        cached = self._make_cached(mid)
        self.assertEqual(ln._local_model_dir(mid), cached)

    def test_missing_cache_returns_model_id(self):
        mid = "iic/not_downloaded_model"
        self.assertEqual(ln._local_model_dir(mid), mid)

    def test_incomplete_cache_without_configuration_returns_model_id(self):
        # 只有目录没有 configuration.json 视为下载不完整,不使用
        mid = "iic/half_downloaded_model"
        self._make_cached(mid, with_config=False)
        self.assertEqual(ln._local_model_dir(mid), mid)

    def test_real_paraformer_cache_resolves_when_present(self):
        # 本机装有缓存时必须命中本地路径(绕过 setUp 的临时 hub)
        mid = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        real = os.path.join(self._old_hub_value(), *mid.split("/"))
        if os.path.isfile(os.path.join(real, "configuration.json")):
            ln.MODELSCOPE_HUB_DIR = self._old_hub
            try:
                p = ln._local_model_dir(mid)
            finally:
                ln.MODELSCOPE_HUB_DIR = self._tmp_name
            self.assertTrue(os.path.isabs(p))
            self.assertNotEqual(p, mid)

    def _old_hub_value(self):
        return os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models")


if __name__ == "__main__":
    unittest.main()
