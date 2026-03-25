#!/usr/bin/env python3
"""Interactive launcher for marker_to_pos services.

Keys:
  j / ↓       move selection down
  k / ↑       move selection up
  r           restart selected service
  s           start selected service
  x           stop selected service
  q / ctrl+c  quit (stops all services)
"""

import subprocess
import threading
import webbrowser
from collections import deque
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, Log, ListView, ListItem
from textual.containers import Vertical

ROOT = Path(__file__).parent

SERVICES = [
    {
        "name": "QR WebSocket Server",
        "cmd": ["uv", "run", "python", "-m", "marker_to_pos.server"],
        "hint": "ws://localhost:8765",
    },
    {
        "name": "Web UI (Flask)",
        "cmd": ["uv", "run", "python", "web/app.py"],
        "hint": "http://localhost:5000",
    },
]

LOG_MAXLEN = 500


class Service:
    def __init__(self, name: str, cmd: list[str], hint: str):
        self.name = name
        self.cmd = cmd
        self.hint = hint
        self.process: subprocess.Popen | None = None
        self._log: deque[str] = deque(maxlen=LOG_MAXLEN)
        self._lock = threading.Lock()
        self._on_line: list = []  # callbacks

    def on_line(self, cb) -> None:
        self._on_line.append(cb)

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            self.cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def restart(self) -> None:
        self.stop()
        self._emit("--- restarting ---")
        self.start()

    @property
    def status(self) -> str:
        if self.process is None:
            return "stopped"
        rc = self.process.poll()
        return "running" if rc is None else f"exited({rc})"

    def get_log(self) -> list[str]:
        with self._lock:
            return list(self._log)

    def _emit(self, line: str) -> None:
        with self._lock:
            self._log.append(line)
        for cb in self._on_line:
            try:
                cb(line)
            except Exception:
                pass

    def _drain(self) -> None:
        assert self.process and self.process.stdout
        try:
            for line in self.process.stdout:
                self._emit(line.rstrip())
        except ValueError:
            pass  # pipe closed by stop()


# ──────────────────────────────────────────────────────────────────────────────


class ServiceItem(ListItem):
    """A list item that displays service name, status, and hint."""

    DEFAULT_CSS = """
    ServiceItem {
        height: 1;
        padding: 0 1;
    }
    ServiceItem Label {
        width: 1fr;
    }
    """

    def __init__(self, service: Service) -> None:
        super().__init__()
        self.service = service

    def compose(self) -> ComposeResult:
        svc = self.service
        yield Label(
            f"{svc.name}  [{svc.status}]  {svc.hint}",
            id=f"label-{id(svc)}",
        )

    def refresh_status(self) -> None:
        svc = self.service
        status = svc.status
        color = "green" if status == "running" else "red"
        self.query_one(Label).update(
            f"{svc.name}  ([{color}]{status}[/{color}])  [dim]{svc.hint}[/dim]"
        )


class LauncherApp(App):
    TITLE = "marker_to_pos launcher"
    BINDINGS = [
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("r", "restart", "Restart"),
        Binding("s", "start", "Start"),
        Binding("x", "stop", "Stop"),
        Binding("o", "open_browser", "Open UI"),
        Binding("q", "quit", "Quit"),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }
    #service-list {
        height: auto;
        max-height: 10;
        border: solid $primary;
        border-title-color: $primary;
    }
    #log-panel {
        border: solid $panel;
        border-title-color: $panel;
        height: 1fr;
    }
    """

    selected: reactive[int] = reactive(0)

    def __init__(self, services: list[Service]) -> None:
        super().__init__()
        self.services = services

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            lv = ListView(*[ServiceItem(s) for s in self.services], id="service-list")
            lv.border_title = "Services"
            yield lv
            log = Log(id="log-panel", highlight=True)
            log.border_title = "Logs"
            yield log
        yield Footer()

    def on_mount(self) -> None:
        for svc in self.services:
            svc.on_line(lambda line, s=svc: self._on_service_line(s, line))
            svc.start()
        self._refresh_items()
        self.set_interval(1.0, self._refresh_items)

    def _on_service_line(self, svc: Service, line: str) -> None:
        if self.services.index(svc) == self.selected:
            try:
                self.call_from_thread(self._append_log, line)
            except Exception:
                pass

    def _append_log(self, line: str) -> None:
        try:
            self.query_one(Log).write_line(line)
        except Exception:
            pass

    def _refresh_items(self) -> None:
        for item in self.query(ServiceItem):
            item.refresh_status()

    def _reload_log(self) -> None:
        log = self.query_one(Log)
        log.clear()
        for line in self.services[self.selected].get_log():
            log.write_line(line)

    # ── actions ──────────────────────────────────────────────────────────

    def action_cursor_down(self) -> None:
        self.query_one(ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(ListView).action_cursor_up()

    def action_restart(self) -> None:
        self.services[self.selected].restart()

    def action_start(self) -> None:
        self.services[self.selected].start()

    def action_stop(self) -> None:
        self.services[self.selected].stop()

    def action_open_browser(self) -> None:
        web_svc = next((s for s in self.services if s.hint.startswith("http")), None)
        if web_svc:
            webbrowser.open(web_svc.hint)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            self.selected = self.query(ServiceItem).nodes.index(event.item)
            self._reload_log()

    def on_unmount(self) -> None:
        for svc in self.services:
            svc.stop()


def main() -> None:
    services = [Service(cfg["name"], cfg["cmd"], cfg["hint"]) for cfg in SERVICES]
    LauncherApp(services).run()


if __name__ == "__main__":
    main()
