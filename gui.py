"""GUI to run the QR detection WebSocket server with configurable address and port."""

import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox

DEFAULT_ADDRESS = "0.0.0.0"
DEFAULT_PORT = 8765


class ServerGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("QR Detection Server")
        self.root.resizable(False, False)

        self.process: subprocess.Popen | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        # Address
        ttk.Label(main, text="Address:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.address_var = tk.StringVar(value=DEFAULT_ADDRESS)
        self.address_entry = ttk.Entry(main, textvariable=self.address_var, width=30)
        self.address_entry.grid(row=1, column=0, sticky=tk.EW, pady=(0, 15))

        # Port
        ttk.Label(main, text="Port:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        self.port_entry = ttk.Entry(main, textvariable=self.port_var, width=30)
        self.port_entry.grid(row=3, column=0, sticky=tk.EW, pady=(0, 20))

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=4, column=0, sticky=tk.EW)

        self.start_btn = ttk.Button(btn_frame, text="Start Server", command=self._start_server)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(
            btn_frame, text="Stop Server", command=self._stop_server, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)

        # Status
        self.status_var = tk.StringVar(value="Server stopped")
        ttk.Label(main, textvariable=self.status_var, foreground="gray").grid(
            row=5, column=0, sticky=tk.W, pady=(15, 0)
        )

        main.columnconfigure(0, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_server(self) -> None:
        address = self.address_var.get().strip() or DEFAULT_ADDRESS
        try:
            port = int(self.port_var.get().strip() or str(DEFAULT_PORT))
        except ValueError:
            messagebox.showerror("Invalid Port", "Please enter a valid port number.")
            return

        if port < 1 or port > 65535:
            messagebox.showerror("Invalid Port", "Port must be between 1 and 65535.")
            return

        cmd = [
            sys.executable,
            "-m",
            "qr_to_pos.server",
            "--host",
            address,
            "--port",
            str(port),
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server: {e}")
            return

        self.status_var.set(f"Server running at ws://{address}:{port}")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.address_entry.config(state=tk.DISABLED)
        self.port_entry.config(state=tk.DISABLED)

    def _stop_server(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None
        self.status_var.set("Server stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.address_entry.config(state=tk.NORMAL)
        self.port_entry.config(state=tk.NORMAL)

    def _on_close(self) -> None:
        if self.process is not None:
            self._stop_server()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ServerGUI()
    app.run()


if __name__ == "__main__":
    main()
