# Signatus Startup Procedure

This procedure starts Signatus on Linux or WSL. The supported CLI supervisor
keeps the AI Service, Core, and GUI as separate processes while coordinating
their ordered startup, readiness, logs, and shutdown.

## 1. One-time environment setup

From the repository root:

```bash
cd /home/signatus/Signatus
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ai,gui,dev]'
cp .env.example .env
```

Do not overwrite an existing `.env`. Before enabling tracking, confirm that its
model paths, camera source, person class, and PPE association match the approved
deployment configuration.

The expected camera settings for the first Linux video device are:

```dotenv
SIGNATUS_CAMERA_SOURCE=/dev/video0
SIGNATUS_AI_TRACKING_ENABLED=true
SIGNATUS_PPE_ASSOCIATION=single_person_frame
SIGNATUS_FRAME_SHM_ENABLED=true
```

## 2. Configure the camera (it may be connected later)

### Native Linux

Connect the approved USB camera and verify that Linux created a video device:

```bash
ls -l /dev/video*
```

With `SIGNATUS_CAMERA_SOURCE=/dev/video0`, Signatus opens the first Linux video
device directly. If the approved camera has a different device path, update the
camera source in `.env`.

### Windows with WSL

Keep the WSL distribution running. In Windows PowerShell, list USB devices:

```powershell
usbipd list
```

Find the camera's current `BUSID`. If its state is not `Shared`, run the
following once from an Administrator PowerShell window:

```powershell
usbipd bind --busid <BUSID>
```

Attach the shared camera to WSL:

```powershell
usbipd attach --wsl --busid <BUSID>
```

For example, the Integrated Camera was detected as bus ID `2-8` when this
procedure was written. Always check `usbipd list` because the bus ID can change.

Back in WSL, verify the result:

```bash
ls -l /dev/video*
```

If camera access is denied, add the current user to the `video` group, then
close and reopen WSL:

```bash
sudo usermod -aG video "$USER"
```

Close Windows Camera, Teams, browsers, and any other application that may be
using the camera. The Signatus AI Service must be the sole camera owner.

Camera presence is not required to start Signatus. The stack starts in a safe
standby state with Camera `STOPPED`; opening the configured device occurs only
after the operator presses **Start Camera** in the GUI.

## 3. Run deployment preflight

From the repository root:

```bash
signatus-launch --check-only
```

This does not open the camera, load inference models, or start any process. It
prints one of `PASS`, `PASS WITH DATA ERRORS`, or `FAIL`, with separate Fatal
errors, Data errors, and Warnings sections. Fatal errors must be fixed. Data
errors disable only affected workers/worksites and are shown again in one GUI
summary. Camera absence is a warning because startup uses Camera `STOPPED`.

## 4. Start Signatus

For the full-screen HDMI display:

```bash
signatus-launch
```

For a development window:

```bash
signatus-launch --windowed
```

The launcher initializes AI models without opening the camera, starts Core,
waits for the AI event connection, and then starts the GUI. Select an available
Wo.No. and press **Start Camera** to begin capture, YOLO inference, and preview.
Press **Stop Camera** to return to standby while all services and models remain
running. Closing the GUI or pressing `Ctrl+C` stops GUI, Core, and AI in that
order. Combined rotating logs are written to `.signatus/logs/signatus.log`.

## 5. Manual diagnostic startup

Use the following only when diagnosing an individual component. Keep the three
processes in separate terminals and start them in order.

### AI Service (terminal 1)

```bash
cd /home/signatus/Signatus
source .venv/bin/activate
python -m uvicorn signatus_ai.app:app \
  --host 127.0.0.1 \
  --port 8001 \
  --env-file .env
```

Wait for model initialization to complete. The camera remains stopped. From
another terminal, the health endpoint can be checked with:

```bash
curl --fail http://127.0.0.1:8001/health
```

### Core (terminal 2)

```bash
cd /home/signatus/Signatus
source .venv/bin/activate
python -m uvicorn signatus_core.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --env-file .env
```

Check Core from another terminal:

```bash
curl --fail http://127.0.0.1:8000/api/health
```

### GUI (terminal 3)

For a development window:

```bash
cd /home/signatus/Signatus
source .venv/bin/activate
set -a
source .env
set +a
signatus-gui --windowed
```

For the full-screen HDMI display, omit `--windowed`:

```bash
signatus-gui
```

Select the required worksite in the GUI, press **Start Camera**, and confirm that
the preview is updating and that the permanent `ONE PERSON AT A TIME` warning is visible.
Preview availability does not alter screening decisions; outcomes continue to
come only from Core.

## 6. Stop manually started Signatus

Close the GUI, then press `Ctrl+C` in the Core and AI Service terminals. Track
state is intentionally cleared whenever Core stops.

For WSL, the camera can then be detached from Windows PowerShell:

```powershell
usbipd detach --busid <BUSID>
```

## Troubleshooting

- No `/dev/video0`: Signatus can remain in Camera `STOPPED`; attach the camera
  to WSL and inspect `usbipd list` before starting capture.
- Camera busy: close other camera applications and retry **Start Camera**. A
  camera `ERROR` does not terminate the service stack.
- Permission denied: verify membership with `getent group video` and reopen WSL
  after adding the user.
- GUI has no preview: verify the AI Service is running on the same Linux/WSL
  host and both processes use the same `SIGNATUS_FRAME_SHM_NAME`.
- Core has no AI events: verify ports `8001` and `8000`, then check both health
  endpoints and terminal logs.
- Tracking is disabled: confirm `SIGNATUS_AI_TRACKING_ENABLED=true` only after
  all approved model, class-policy, camera, and association settings are ready.
