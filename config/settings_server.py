"""
設定頁面本地伺服器
啟動一個小型 HTTP 伺服器，讓使用者透過瀏覽器管理設定
提供 REST API 讀寫 config.json
"""

import json
import logging
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.llm import LLMProcessor

logger = logging.getLogger("VoiceType.SettingsServer")

_server_thread = None
_server_instance = None


class SettingsAPIHandler(SimpleHTTPRequestHandler):
    """處理設定 API 和靜態檔案"""

    settings = None  # 由外部注入

    def __init__(self, *args, **kwargs):
        self.directory = str(Path(__file__).parent.parent / "ui")
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/config":
            # 回傳當前設定
            self._send_json(self.settings.get_config())
        elif parsed.path == "/api/models":
            query = parse_qs(parsed.query)
            provider = query.get("provider", [""])[0]
            if not provider:
                self._send_json({"status": "error", "message": "provider is required"}, code=400)
                return

            try:
                llm = LLMProcessor(self.settings)
                models = llm.list_models(provider=provider, cfg=self.settings.get_config())
                self._send_json({"status": "ok", "provider": provider, "models": models})
            except Exception as e:
                logger.warning("取得模型清單失敗 (%s): %s", provider, e)
                self._send_json({"status": "error", "message": str(e), "models": []}, code=502)
        elif parsed.path == "/api/health":
            self._send_json({"status": "ok", "version": "0.1.0"})
        else:
            # 靜態檔案（UI）
            if parsed.path == "/" or parsed.path == "":
                self.path = "/settings.html"
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/config":
            # 更新設定
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                new_config = json.loads(body.decode("utf-8"))
                self.settings.update_all(new_config)
                self._send_json({"status": "ok", "message": "設定已儲存"})
                logger.info("設定已透過 Web UI 更新")
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, code=400)

        elif parsed.path == "/api/config/key":
            # 更新單一 API Key
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                provider = data.get("provider")
                key = data.get("key", "")
                self.settings.set_api_key(provider, key)
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, code=400)
        else:
            self._send_json({"error": "Not Found"}, code=404)

    def do_OPTIONS(self):
        """處理 CORS 預檢請求"""
        self.send_response(200)
        self._add_cors_headers()
        self.end_headers()

    def _send_json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _add_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        """靜默 HTTP 請求日誌（太多雜訊）"""
        pass


def start_settings_server(settings, port=18923):
    """啟動設定伺服器（背景執行緒）"""
    global _server_thread, _server_instance

    if _server_thread and _server_thread.is_alive():
        # 已經在跑了，直接開瀏覽器
        webbrowser.open(f"http://localhost:{port}")
        return

    SettingsAPIHandler.settings = settings

    try:
        _server_instance = HTTPServer(("127.0.0.1", port), SettingsAPIHandler)
    except OSError:
        # 端口被佔用，可能上次沒關乾淨
        logger.warning("Port %d 被佔用，直接開啟瀏覽器", port)
        webbrowser.open(f"http://localhost:{port}")
        return

    def run_server():
        logger.info("設定伺服器已啟動: http://localhost:%d", port)
        _server_instance.serve_forever()

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

    webbrowser.open(f"http://localhost:{port}")


def stop_settings_server():
    """停止設定伺服器"""
    global _server_instance
    if _server_instance:
        _server_instance.shutdown()
        _server_instance = None
