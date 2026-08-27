"""
OmniScan 3D — One-Click Launcher
Detects local IP, prints terminal QR-code for quick mobile connection, and launches FastAPI server.
"""

import os
import sys
import socket
import qrcode
import uvicorn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_banner(local_ip, port=8000):
    url = f"http://{local_ip}:{port}"
    localhost_url = f"http://127.0.0.1:{port}"

    print("\n" + "=" * 62)
    print("       🚀 OmniScan 3D — Server & Web Dashboard Pornit      ")
    print("=" * 62)
    print(f"\n  💻 Acces de pe Laptop / PC:  \033[1;36m{localhost_url}\033[0m")
    print(f"  📱 Acces de pe Telefon/Mobil: \033[1;32m{url}\033[0m\n")
    print("  Scanează codul QR de mai jos cu camera telefonului (în aceeași rețea Wi-Fi):")
    print("-" * 62)

    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        pass

    print("-" * 62)
    print("  Apasă Ctrl + C pentru a opri serverul.\n")


def main():
    port = 8000
    local_ip = get_local_ip()
    print_banner(local_ip, port)

    # Launch uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
