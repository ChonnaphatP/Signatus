# Signatus Developer Instructions

## Role

Act as the implementation and verification developer for the approved Signatus v1 design.
Preserve its safety boundaries, typed contracts, and fail-closed behavior. Keep changes focused
on the requested work, test affected behavior, and report any missing project data or architecture
decision instead of inventing it.

## Approved architecture

Signatus v1 has three components:

```text
PySide6 GUI <-> Python Core <-> Python AI Service <-> USB camera
      ^                              |
      +----- shared-memory frames ---+
```

- The AI Service captures camera frames and runs Ultralytics YOLO OpenVINO tracking with
  ByteTrack. It publishes `PERSON_SEEN` and `TRACK_LOST` events to Core over WebSocket and
  serves face-embedding and latest-cached-PPE commands over REST. It is the sole camera owner
  and publishes raw BGR preview frames through the approved local double-buffered shared memory.
- Core owns all screening state, identity matching, PPE policy evaluation, authorization rules,
  worksite selection, and final outcomes. Track records live only in a Python dictionary and are
  cleared on `TRACK_LOST` or Core restart.
- The PySide6 GUI is display-only. It selects and reads worksite/state through Core REST and
  receives `AUTHORIZED`, `PPE_VIOLATION`, `UNAUTHORIZED`, or `FACE_CAPTURE_FAILED` from Core
  over WebSocket. It reads presentation-only frames from shared memory and permanently displays
  `ONE PERSON AT A TIME`.
- Face capture retries use a one-second cooldown. Three consecutive failures mark the track
  handled until exit. A completed identity/PPE decision also marks the track handled.
- PPE evaluation uses the owner-approved Core mapping from semantic worksite PPE names to exact
  model positive and negative class names. A negative observation wins; a missing observation is
  non-compliant; `none` marks every required item missing. A non-OK PPE result is treated as no
  detections and therefore fails closed.
- AI Service uses the owner-approved OpenCV YuNet detector and SFace FP32 ONNX
  descriptor model. Core owns cosine-similarity matching.

## Scope limits

- Do not move identity, PPE, or authorization decisions into the GUI or AI Service.
- Do not add persistent track state; v1 track state is process-memory-only.
- Do not replace cached PPE command behavior with an unapproved transport or workflow.
- Preserve the approved `single_person_frame` v1 association. Frames without exactly the requested
  single tracked person must return `ASSOCIATION_UNRESOLVED` and fail closed.
- Treat `0.35` as the owner-approved functional threshold, but not as production-validated until
  the representative threshold experiment is complete.
- Do not invent camera configuration or replacement face backends when project data is absent.
- Do not replace the approved local shared-memory preview with MJPEG, WebRTC, a Core proxy, direct
  GUI camera ownership, or another transport without an architecture decision.
- Preserve the four distinct GUI outcomes. In particular, do not represent
  `FACE_CAPTURE_FAILED` as `UNAUTHORIZED`.

## Parked decisions

Parked decisions require explicit owner approval before implementation or enablement.
IoU/multi-person PPE association, any replacement for the approved shared-memory preview, any
replacement face backend, and any new interface or process-boundary change require approval.
The final production matching threshold remains pending the approved experiment. Record the
blocker and continue only with work that does not prejudge the decision.

## Test and verification commands

Set up the supported environment when needed:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ai,gui,dev]'
```

Run the canonical safety-critical domain suite, which does not load the model:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The equivalent configured pytest suite and lint check are:

```bash
python -m pytest
python -m ruff check .
```

When changing AI integration, also verify model loading and the relevant camera/runtime path in
the deployment environment. Tracking stays disabled until the OpenVINO model directory, exact
class policy, camera source, and owner-approved association strategy are all ready.
