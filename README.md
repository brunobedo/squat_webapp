<div align="center">

# Squat Webapp

**Markerless 2D squat analysis in the browser — hip and knee kinematics from a single sagittal-view video.**

[![Live app](https://img.shields.io/badge/Live%20app-squatweb.streamlit.app-black?logo=streamlit&logoColor=white)](https://squatweb.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-black?logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-black?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Paper](https://img.shields.io/badge/DOI-10.3390%2Fbiomechanics6010001-black)](https://doi.org/10.3390/biomechanics6010001)

*Built for biomechanists. Minimalist, black & white, in Portuguese (pt-BR).*

**Try it now → [squatweb.streamlit.app](https://squatweb.streamlit.app)**

</div>

---

## Overview

Squat Webapp is a web interface for the markerless pipeline from
[mediapipe2dangle](https://github.com/brunobedo/mediapipe2dangle). Upload a squat video
recorded from the side (sagittal plane) and the app runs MediaPipe Pose on every frame,
computes 2D hip and knee flexion angles, and gives you:

| | Feature | Description |
|---|---|---|
| 1 | **Video input** | Upload your own video (`mp4`, `mov`, `avi`, `mkv`) or pick a bundled sample from `videos/`. Choose the body side facing the camera and, optionally, the MediaPipe confidence threshold. |
| 2 | **Synchronized analysis** | The processed video (skeleton + angle overlays, H.264) plays side by side with the angle-vs-frame chart. The chart cursor follows playback in real time; clicking a point on the chart seeks the video to that instant. |
| 3 | **Squat metrics** | Per-joint table: peak flexion (and its timestamp), minimum flexion, range of motion, peak angular velocity in flexion and extension, and mean angular velocity. |
| 4 | **Export** | One-click download of the angle time series (`.csv`) and the processed video (`.mp4`). |

## How it works

```
video (sagittal view)
        │
        ▼
MediaPipe Pose  ──►  shoulder / hip / knee / ankle landmarks (per frame)
        │
        ▼
2D joint angles  ──►  knee = 180° − angle(hip, knee, ankle)
                      hip  = 180° − angle(shoulder, hip, knee)
        │
        ├──►  processed video (skeleton + angles, H.264 via ffmpeg)
        ├──►  CSV (Frame, Knee_Angle, Hip_Angle)
        └──►  metrics (peaks, ROM, angular velocities)
```

Frames without a detected pose are stored as `NaN` in the CSV (keeping frame indices
aligned with the video) and linearly interpolated only for the velocity computation.

## Using the app

No installation required — the app is live at
**[squatweb.streamlit.app](https://squatweb.streamlit.app)**. Pick a sample video
(or upload yours), choose the body side facing the camera, and click
**Processar vídeo**.

### Recording recommendations

- Camera perpendicular to the sagittal plane (true side view), tripod if possible.
- Whole body visible during the entire movement — shoulder, hip, knee and ankle.
- Good lighting and contrast between the subject and the background.

## Running locally

```bash
git clone https://github.com/brunobedo/squat_webapp.git
cd squat_webapp

# create an environment (conda shown; venv works too)
conda create -n squat_webapp python=3.11
conda activate squat_webapp

pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

### Command-line usage

The original batch pipeline is still available:

```bash
python run.py --videopath ./videos/video_test_1.mp4 --side r --save True --min_confidence 0.8
```

Results (video, CSV and plot) are written to `videos/results/<video_name>/`.

## Project structure

```
squat_webapp/
├── app.py                # Streamlit webapp (UI + processing + synced player)
├── run.py                # CLI pipeline (batch processing)
├── src/
│   └── maintools.py      # angle calculation and plotting helpers
├── videos/               # sample squat videos
├── .streamlit/
│   └── config.toml       # black & white theme, 500 MB upload limit
└── requirements.txt
```

## Deployment notes

The app runs on Streamlit Community Cloud at
[squatweb.streamlit.app](https://squatweb.streamlit.app). If you deploy your own fork:

- Select **Python 3.11** in "Advanced settings" when creating the app — `mediapipe`
  has no wheels for the newest Python versions.
- `requirements.txt` uses `opencv-python-headless`, so no system GUI libraries are
  required.
- `imageio-ffmpeg` ships its own ffmpeg binary — no `packages.txt` needed.

## Citing

If you use this tool in your research, please cite:

> PEREIRA, Dayanne R.; CATELLI, Danilo S.; SANTIAGO, Paulo R. P.; BEDO, Bruno L. S.
> **Markerless Pixel-Based Pipeline for Quantifying 2D Lower Limb Kinematics During
> Squatting: A Preliminary Validation Study**. *Biomechanics*, v. 6, n. 1, p. 1, 2025.
> DOI: [10.3390/biomechanics6010001](https://doi.org/10.3390/biomechanics6010001)

```bibtex
@article{Pereira2025,
  title     = {Markerless Pixel-Based Pipeline for Quantifying 2D Lower Limb Kinematics
               During Squatting: A Preliminary Validation Study},
  author    = {Pereira, Dayanne R. and Catelli, Danilo S. and Santiago, Paulo R. P.
               and Bedo, Bruno L. S.},
  journal   = {Biomechanics},
  volume    = {6},
  number    = {1},
  pages     = {1},
  year      = {2025},
  month     = dec,
  publisher = {MDPI AG},
  issn      = {2673-7078},
  doi       = {10.3390/biomechanics6010001},
  url       = {http://dx.doi.org/10.3390/biomechanics6010001}
}
```

## Acknowledgements

Based on [mediapipe2dangle](https://github.com/brunobedo/mediapipe2dangle) (GPL-3.0).
Partially funded by the Dean's Office for Research and Innovation of the University of
São Paulo and by the São Paulo Research Foundation (FAPESP, grant #2024/10736-9).
