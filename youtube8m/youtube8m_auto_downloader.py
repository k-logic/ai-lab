"""
YouTube動画からフレームを抽出して PNG で保存するスクリプト
キャリブレーション用に高画質な 1280x720 画像を大量に生成可能

=== 必要なもの ===
Python 標準ライブラリのみ使用 (os, subprocess, pathlib)
外部ツール:
  - yt-dlp (YouTubeダウンローダー)
      pip install yt-dlp
      または brew install yt-dlp (macOS)
  - ffmpeg (動画 → 画像フレーム抽出)
      brew install ffmpeg        # macOS
      sudo apt install ffmpeg    # Ubuntu/Debian
      choco install ffmpeg       # Windows (Chocolatey)
"""

import os
import subprocess
from pathlib import Path

# === 設定 ===
video_ids = [
    "qogIVb8UGWM",
    "rUWxSEwctFU"
]
save_root = Path("youtube8m_calib")
video_dir = save_root / "videos"
frames_dir = save_root / "frames"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

# === ダウンロード + フレーム抽出 ===
for vid in video_ids:
    url = f"https://www.youtube.com/watch?v={vid}"
    video_base = video_dir / vid  # 拡張子なし
    print(f"[INFO] Downloading: {url}")

    # 実際の拡張子付きで保存
    subprocess.run([
        "yt-dlp", "-q",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "-o", str(video_base) + ".%(ext)s",
        url
    ])

    # ダウンロードされたファイルを探す
    candidates = list(video_dir.glob(f"{vid}.*"))
    if not candidates:
        print(f"[WARN] Failed to download {vid}, skipping...")
        continue

    video_path = candidates[0]
    print(f"[INFO] Extracting frames from: {video_path}")

    # PNG出力（非圧縮、劣化なし）
    output_pattern = frames_dir / f"{vid}_%06d.png"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", "fps=1,scale=1280:720",
        str(output_pattern)
    ])

print("\n✅ 完了！キャリブレーション用PNG画像は以下に保存されました:")
print(f"📁 {frames_dir.resolve()}")

