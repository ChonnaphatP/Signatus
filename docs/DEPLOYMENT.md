# Reproducible Linux deployment

## Supported deployment envelope

The repository-supported target is a native x86-64 Linux host with:

- CPython 3.11, 3.12, 3.13, or 3.14;
- network access to PyPI during installation, or an owner-maintained wheelhouse containing every
  pinned dependency;
- a POSIX shared-memory filesystem at `/dev/shm`;
- a local X11, Wayland, EGLFS, LinuxFB, or VKKHR display backend for the GUI;
- a V4L2-compatible USB camera available exclusively to the Signatus AI Service when screening;
- enough storage for Python, OpenVINO, PySide6, Ultralytics/PyTorch, and the model artifacts.

Other operating systems and CPU architectures are not deployment-validated by this repository.

## Fresh-clone installation

```bash
git clone https://github.com/ChonnaphatP/Signatus.git
cd Signatus
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` installs the package plus its pinned AI and GUI dependency groups. Development
tools remain optional; install them with `.venv/bin/python -m pip install -e '.[dev]'` when needed.
Use a new virtual environment. Installing more than one OpenCV wheel distribution in the same
environment is unsupported because they all provide the same `cv2` package.

## Required non-Git assets

Copy the five exact model files listed in [`models/README.md`](../models/README.md) into `models/`.
They are omitted from Git because they are binary deployment assets with separate licensing and
distribution obligations. Copy owner-approved worksite and worker-profile data separately; those
files contain biometric or identifying data and must not enter source control.

Edit `.env` only after confirming the deployment box. At minimum:

```dotenv
SIGNATUS_CAMERA_SOURCE=/dev/video0
SIGNATUS_AI_TRACKING_ENABLED=true
```

Do not assume `/dev/video0` on another host. Select the actual exclusive camera device. Preserve
the approved model paths, exact `Person` class, `single_person_frame` association, shared-memory
preview, and provisional `0.35` threshold unless the owner approves an architecture change.

## Acceptance gate

Run these commands on the target machine from the repository root:

```bash
.venv/bin/signatus-launch --check-only
.venv/bin/python -m pip check
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Preflight must report `RESULT: PASS` or `RESULT: PASS WITH DATA ERRORS`. Fix every `FATAL` item.
Data errors disable only the affected worker or worksite; they are not approval to use invalid data.
Camera absence is a warning because the stack starts with Camera `STOPPED`.

Then perform the hardware acceptance test:

1. Start `.venv/bin/signatus-launch` on the physical display host.
2. Confirm AI and Core become ready while Camera remains `STOPPED`.
3. Select an owner-approved available worksite and start the camera.
4. Confirm live preview, same-frame overlays, face capture, PPE outcomes, camera stop/start, and
   ordered shutdown on the actual camera, display, lighting, and Intel deployment hardware.

Repository checks cannot replace this physical acceptance test or the pending representative
face-threshold experiment. Until both succeed, the build is deployable for validation but is not a
production-validated screening release.
