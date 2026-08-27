#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# OmniScan 3D — Script Automatizare Termux (Android)
# Sincronizare fotografii de pe telefon -> Google Drive -> Descărcare Model 3D
# ==============================================================================

echo "=================================================="
echo "          OmniScan 3D — Termux Mobile Hub         "
echo "=================================================="

# Verificare dependențe
if ! command -v rclone &> /dev/null; then
    echo "[!] Se instalează rclone în Termux..."
    pkg update -y && pkg install -y rclone
fi

REMOTE_NAME="gdrive"
REMOTE_INPUT="Fotogrammetrie/Input"
REMOTE_OUTPUT="Fotogrammetrie/Output"
LOCAL_IMAGES="/sdcard/DCIM/OmniScan/"
LOCAL_DOWNLOADS="/sdcard/Download/OmniScan3D/"

mkdir -p "$LOCAL_IMAGES"
mkdir -p "$LOCAL_DOWNLOADS"

echo ""
echo "Selectează acțiunea dorită:"
echo "1) Încarcă fotografiile din telefon în Google Drive ($LOCAL_IMAGES -> $REMOTE_INPUT)"
echo "2) Descarcă modelul 3D finalizat din Google Drive ($REMOTE_OUTPUT -> $LOCAL_DOWNLOADS)"
echo "3) Sincronizare completă (Upload + Așteptare + Download)"
read -p "Opțiune [1-3]: " opt

case $opt in
    1)
        echo "[*] Se încarcă fotografiile în Google Drive..."
        rclone copy "$LOCAL_IMAGES" "$REMOTE_NAME:$REMOTE_INPUT" -P
        echo "[+] Upload finalizat cu succes!"
        ;;
    2)
        echo "[*] Se descarcă modelul 3D și rezultatele..."
        rclone copy "$REMOTE_NAME:$REMOTE_OUTPUT" "$LOCAL_DOWNLOADS" -P
        echo "[+] Descărcare finalizată! Modelele sunt în: $LOCAL_DOWNLOADS"
        ;;
    3)
        echo "[*] Se încarcă fotografiile..."
        rclone copy "$LOCAL_IMAGES" "$REMOTE_NAME:$REMOTE_INPUT" -P
        echo "[*] Acum pornește Google Colab pentru procesare."
        read -p "Apasă ENTER după ce Colab a finalizat reconstrucția..."
        echo "[*] Se descarcă rezultatele..."
        rclone copy "$REMOTE_NAME:$REMOTE_OUTPUT" "$LOCAL_DOWNLOADS" -P
        echo "[+] Totul este gata!"
        ;;
    *)
        echo "[X] Opțiune invalidă."
        ;;
esac
