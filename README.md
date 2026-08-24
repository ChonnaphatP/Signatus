# Signatus Software

Signatus v1 is an edge AI entrance-screening system. It identifies a worker,
checks required PPE for the selected worksite, and sends one of four outcomes to
a display-only PySide6 GUI.

This repository implements the approved v1 architecture:

```text
PySide6 GUI <-> Python Core <-> Python AI Service <-> USB camera
      ^                              |
      +----- shared-memory frames ---+
```

- AI Service publishes tracking events over WebSocket.
- Core sends embedding and PPE commands to the AI Service over REST.
- Core publishes outcome signals to the GUI over WebSocket.
- Track handling state stays in RAM and resets when Core stops.
- PPE commands read the latest cached YOLO detection result.
- AI Service owns camera capture and publishes presentation-only BGR frames and
  same-frame YOLO overlay metadata to the local GUI through a versioned,
  double-buffered shared-memory segment.
- AI Service uses OpenCV YuNet face detection and SFace FP32 ONNX descriptors;
  Core performs cosine-similarity identity matching.
- Camera lifecycle commands travel GUI → Core → AI Service. The AI Service
  starts healthy with the camera stopped and keeps all models resident across
  camera stop/start cycles.

## Current build status

Implemented:

- Typed JSON contracts for AI events, AI command results, and GUI signals.
- Core-owned, smallest-scope Wo.No./worker validation, unavailable-worksite
  selection state, and a summarized GUI data-error report.
- Standby and Authorization Core state machine.
- In-memory track guard with 1-second face retry cooldown and 3-attempt limit.
- Cosine-distance identity matching.
- Owner-approved, exact positive and negative PPE class policy.
- Fail-closed `single_person_frame` PPE association and persistent operator warning.
- AI detection cache, WebSocket event hub, and REST command endpoints.
- Ultralytics ByteTrack adapter for an OpenVINO YOLO export.
- Native workstation-style PySide6 GUI with a local live preview and Qt-painted
  YOLO detection overlays for a Linux HDMI display.
- Fail-closed CLI preflight and AI/Core/GUI process supervision with continuous
  operational-readiness monitoring, single-instance protection, launch IDs,
  process-group cleanup, and rotating logs.
- Unit tests for the safety-critical Core decisions.

Still required before a production-stable release:

- Confirm the USB camera source on the deployment box.
- Validate the OpenVINO model, camera, shared-memory preview, and HDMI runtime on
  the i3-8100 deployment hardware.
- Enroll owner-approved face images through the Worker Profile tool; Wo.No.
  Create/Edit generates a real 128-value SFace descriptor through the AI Service
  and stores only that descriptor under `config/worksites/`. Deliberately invalid
  samples live under `examples/worksites/`.
- Calibrate and validate the provisional `0.35` minimum cosine-similarity threshold with
  representative workers, camera placement, and lighting.

## Linux setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ai,gui,dev]'
cp .env.example .env
```

Start the AI Service:

```bash
uvicorn signatus_ai.app:app --host 127.0.0.1 --port 8001 --env-file .env
```

Start Core:

```bash
uvicorn signatus_core.app:app --host 0.0.0.0 --port 8000 --env-file .env
```

Start the full-screen GUI on the machine connected to the HDMI display:

```bash
signatus-gui
```

For normal operation, use the CLI supervisor instead of starting three
terminals manually:

```bash
signatus-launch --check-only
signatus-launch
```

It performs fail-closed deployment checks, starts AI then Core then GUI, waits
for operational readiness, monitors the services, writes rotating logs, and
stops the processes in reverse order. The normal initial state is AI `READY`,
Core `READY`, GUI running, and Camera `STOPPED`; camera frames are not a service
readiness signal. Use `signatus-launch --windowed` for
development or `signatus-launch --no-gui` to supervise only AI and Core. See
`docs/LAUNCHER.md` for the complete contract and options.

The GUI uses `http://127.0.0.1:8000` and shared-memory segment
`signatus_camera_v1` by default. Set `SIGNATUS_CORE_URL` and
`SIGNATUS_FRAME_SHM_NAME`, or use the matching CLI options, to override them.
The GUI and AI Service must run on the same Linux host for live preview. Use
`signatus-gui --windowed` for development.

Preview availability never changes a screening decision. The GUI still receives
all identity, PPE, and authorization outcomes only from Core.

After selecting an available Wo.No., use the GUI's **Start Camera** and **Stop
Camera** controls. Stop releases the physical device and invalidates cached
detections/preview data without stopping Core, GUI, AI, or reloading models.

Tracking is disabled in `.env.example`. Set `SIGNATUS_AI_TRACKING_ENABLED=true`
only after confirming the model directory and camera source. The default preview
slot capacity is 6,220,800 bytes, enough for raw 1920x1080 BGR; increase
`SIGNATUS_PREVIEW_MAX_FRAME_BYTES` before startup for larger frames.

## Tests

The domain tests use Python's standard library and do not load the model:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Windows copy target

After extracting this repository, copy or clone it to:

```text
E:\Workspace\โครงงานปี 3\Signatus Software Repo
```

The Linux deployment must use Linux paths in `.env`; the Windows paths are
development source locations only.

## Documents

- `docs/ARCHITECTURE.md` records the implemented process and transport boundaries.
- `docs/DEVELOPER_REPORT.md` lists parked decisions and the data still needed.
- `docs/LAUNCHER.md` describes one-command preflight, startup, supervision, and
  shutdown.
