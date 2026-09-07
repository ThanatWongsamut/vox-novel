# VoxNovel Importer - Chrome Extension

1-Click extension to import web novel chapters (ReadToon, WebNovel, etc.) directly into VoxNovel for offline reading and **VoxCPM2 Thai Audiobook Synthesis**.

---

## 🚀 Quick Setup (30 Seconds)

1. Open **Google Chrome** and navigate to `chrome://extensions/`.
2. Toggle **Developer mode** on (top-right corner).
3. Click **Load unpacked** (top-left).
4. Select this directory:
   ```
   /Users/titor/Work/Project/vox-novel/extension
   ```
5. Done! The **VoxNovel Importer** icon (`🎙️`) will appear in your Chrome toolbar.

---

## 📖 How to Use

1. Make sure VoxNovel local server is running:
   ```bash
   uv run vox-novel web
   ```
2. In Google Chrome, go to any unlocked chapter on [ReadToon](https://readtoon.com) (e.g. `https://readtoon.com/content/novel-mypossessionbecameaghoststory/170`).
3. Click the floating **[🎙️ Import to VoxNovel]** button in the bottom-right corner (or click the extension icon in the toolbar).
4. VoxNovel will instantly extract the decrypted prose, save it locally, and automatically open the **VoxNovel Reader** ready for reading or generating voice audio!
