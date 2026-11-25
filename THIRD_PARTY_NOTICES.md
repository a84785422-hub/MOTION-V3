# Third-party software

The Windows build of Video Motion Control bundles the following components.

| Component | Use | License |
|-----------|-----|---------|
| [FFmpeg](https://ffmpeg.org) (`ffmpeg.exe`, `ffprobe.exe`) | decoding, encoding, probing | GPL v3 for the builds fetched by `scripts/fetch_ffmpeg.ps1` (they include libx264 and libx265) |
| [PySide6 / Qt for Python](https://doc.qt.io/qtforpython-6/) | graphical interface | LGPL v3 |
| [NumPy](https://numpy.org) | array maths | BSD 3-Clause |
| [OpenCV](https://opencv.org) (`opencv-python-headless`) | image resampling | Apache 2.0 |
| [ONNX Runtime](https://onnxruntime.ai) | exécution des modèles IA sur CPU | MIT |
| [LivePortrait](https://github.com/KwaiVGI/LivePortrait) / [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) | algorithme et conversion ONNX portés dans `src/vmc/ai` ; `mask_template.png` | MIT |
| Poids ONNX (téléchargés au premier lancement, non embarqués) | LivePortrait : MIT ; détection de visage InsightFace (`retinaface_det_static`, `face_2dpose_106_static`) : **non commercial** | voir README |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | détection de pose (mode corps entier) ; modèle `pose_landmarker_full.task` téléchargé au premier lancement | Apache 2.0 |
| [PyInstaller](https://pyinstaller.org) | packaging | GPL v2 with a bootloader exception that allows distributing the resulting programs under any license |

## Distributing the executable

* The source code of this project is MIT licensed (see `LICENSE`).
* The FFmpeg binaries fetched by default are **GPL** builds. If you redistribute a build that
  contains them, you must comply with the GPL for those binaries: keep their license text (it is
  copied next to them in `ffmpeg/LICENSE.txt`) and be ready to provide the corresponding FFmpeg
  source, which the FFmpeg build providers publish.
* To ship under permissive terms only, point `scripts/fetch_ffmpeg.ps1 -Url` at an LGPL FFmpeg
  build. The H.264 / H.265 (software) presets then need an encoder that build includes; ProRes,
  FFV1 and hardware (NVENC) presets do not depend on GPL components.
* Qt is used through dynamic linking in a one-folder build, which satisfies the LGPL: users can
  replace the Qt libraries in the application folder.
