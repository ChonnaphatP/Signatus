# Signatus v1 Implemented Architecture

## Process boundary

```text
PySide6 GUI
  | REST: worksite selection, validation, state, camera start/stop
  | WebSocket: outcome signals
Python Core
  | REST: camera lifecycle, embedding, and cached PPE commands
  | WebSocket: person-seen and track-lost events
Python AI Service
  | OpenCV USB capture
  | Ultralytics YOLO OpenVINO tracking with ByteTrack
Camera

Python AI Service -- local shared-memory BGR preview --> PySide6 GUI
```

The GUI never evaluates identity or PPE. The AI Service never decides whether a
worker is authorized or compliant. Core owns all state and rules.
The GUI never opens the camera: start/stop commands always travel through Core,
and the AI Service remains the sole camera/resource/inference owner.

The separate `signatus-launch` supervisor owns only process orchestration. It
performs static configuration checks, starts AI then Core then GUI, and shuts
them down in reverse order. It consumes health/readiness metadata but no
identity, PPE, authorization, or tracking-event data. It never restarts Core
automatically because a Core restart intentionally clears in-memory track
state.

The PySide6 process uses Qt's native HTTP and WebSocket clients. Core no longer
hosts browser assets. The existing `/api/worksites`, worksite-selection,
`/api/state`, and `/ws/gui` interfaces remain the decision and state boundary.
The only AI-to-GUI path is an owner-approved, presentation-only local frame
buffer. It carries each raw frame plus that same inference result's class,
confidence, and bounding-box metadata for display. It carries no identity, PPE
evaluation, or authorization result.

## Track lifecycle

Core stores track records in a Python dictionary. It performs no disk writes.

1. `PERSON_SEEN` creates or refreshes an in-memory track record.
2. A new eligible track starts Authorization.
3. A completed identity and PPE decision marks the track handled.
4. A face capture failure leaves the track eligible after a 1-second cooldown.
5. Three consecutive face capture failures mark the track handled.
6. `TRACK_LOST` deletes the full record. Re-entry starts a new screening.
7. Restarting Core clears every track record.

The AI Service emits `TRACK_LOST` after the configured absence timeout. The
initial value is 1.5 seconds and belongs to runtime configuration.

## AI event transport

`GET ws://127.0.0.1:8001/ws/events`

Event examples:

```json
{"type":"PERSON_SEEN","track_id":3,"captured_at":1786980000.125}
```

```json
{"type":"TRACK_LOST","track_id":3,"captured_at":1786980002.410}
```

## Core command transport

Embedding command:

```text
POST /commands/tracks/{track_id}/embedding
```

Successful response shape (the descriptor contains exactly 128 finite floats):

```text
{"track_id":3,"status":"OK","embedding":[<128 finite floats>]}
```

Failure response:

```json
{"track_id":3,"status":"NO_FACE","embedding":null}
```

AI Service crops the requested tracked-person box from the latest cached BGR
frame. OpenCV YuNet must find exactly one face in that crop; SFace then aligns
the face and returns its FP32 ONNX descriptor. No face, multiple faces, invalid
pixels, and invalid descriptors remain distinct fail-closed command results.
Core compares the descriptor with enrolled worker descriptors using cosine
similarity. The current minimum accepted similarity is the owner-approved
functional value `0.35`; production calibration is still required.
AI rejects any descriptor size other than the approved 128-value SFace output,
Core rejects a malformed descriptor response, and launcher preflight rejects
enrolled vectors with another size.

Worker enrollment image command:

```text
POST /commands/faces/embedding
```

```json
{"face_image":"data:image/jpeg;base64,..."}
```

This camera-independent command reuses the AI Service's initialized YuNet and
SFace models and returns the existing embedding-result contract with reserved
`track_id: 0`. Worker Profile files contain only `worker_id`, `name`, and the
face-image data URI. During Wo.No. Create/Edit, Core calls this command and then
atomically stores only `worker_id`, `name`, and the resulting strict 128-value
descriptor in the compact Wo.No. file. Enrollment failures are scoped data
errors; they do not change camera state or AI service health.

PPE cache command:

```text
POST /commands/tracks/{track_id}/ppe
```

Response:

```json
{
  "track_id":3,
  "status":"OK",
  "detected_classes":["helmet","no_gloves","boots"],
  "captured_at":1786980000.125
}
```

Core treats a non-OK PPE result as an empty detection set. The approved missing
detection rule then marks every required but unobserved item non-compliant.

## GUI outcomes

Core publishes these values through `/ws/gui`:

- `AUTHORIZED`
- `PPE_VIOLATION`
- `UNAUTHORIZED`
- `FACE_CAPTURE_FAILED`

`FACE_CAPTURE_FAILED` adds `face_failure_reason`, `attempt`, and
`retry_allowed`. It does not impersonate `UNAUTHORIZED`.

## Camera display

AI Service remains the sole camera owner. It writes the latest raw BGR frame to
two fixed-capacity slots in a `multiprocessing.shared_memory` segment. Preview
contract v2 adds a fixed-capacity metadata slot beside each frame. Its compact
binary records contain only YOLO class, confidence, and source-frame box
coordinates. A 64-byte versioned header (`<8s9IQQ4x`) records the layout,
overlay size, active slot, odd/even seqlock sequence, and capture timestamp.
GUI copies pixels and metadata only when identical even headers surround both
copies; incompatible, malformed, or stale buffers display `Camera unavailable`
without affecting Core signals. QPainter scales boxes through the same
aspect-ratio-preserving image rectangle, including letterbox offsets.

The default segment is `signatus_camera_v1`. AI Service creates and unlinks it;
GUI attaches without ownership. Both processes must run on the same Linux host.
While Camera is `STOPPED`, the owned segment contains an empty header. Stopping
capture invalidates the header and cached detections before returning to safe
standby; starting capture resumes publication without reloading models.
The detection screen permanently displays `ONE PERSON AT A TIME` because the v1
PPE association is valid only for a controlled single-person checkpoint.

## Operational readiness

AI health and camera state are independent. AI `READY` means the service APIs,
YOLO/exact class map, YuNet, SFace, association, and required preview IPC are
initialized. It does not require an open camera, inference task, cached frame,
or published preview. Camera `STOPPED` is the normal startup state; camera open
failure becomes Camera `ERROR` while the AI service remains `READY`. Core health
separately reports whether its AI tracking-event WebSocket is connected.

The launcher waits for these conditions before exposing the GUI and monitors
them afterward. Three consecutive runtime readiness failures stop the stack;
Core is not silently restarted.

## Deployment-data validation

Core owns one cached validation catalog and exposes it to the GUI. Fatal
infrastructure/configuration errors block startup. Invalid workers are removed
from only their Wo.No. authorization set; other valid workers remain usable.
Broken worksite-level PPE configuration disables only that Wo.No. The GUI marks
disabled worksite entries and shows one summarized data-error dialog with
expandable details. Production data remains strict: malformed embeddings are
never padded, truncated, reshaped, replaced, or generated.

## PPE policy boundary

Wo.No. files retain the approved `required_ppe` string list. Core owns the fixed
v1 mapping from `helmet`, `gloves`, `vest`, `boots`, and `goggles` to the exact
approved model classes. The model's person class is exactly `Person`.

When both positive and negative classes appear, the negative class wins. When
neither appears, the item is missing and the worker is non-compliant. Vest has no
negative class, so it passes only when `vest` is observed. Detection of `none`
means complete PPE absence and marks every required item missing, even if a
positive class is also present.

`single_person_frame` returns PPE observations only when the requested track is
the sole tracked person in the cached frame. Zero-person, multiple-person, and
track-mismatch frames return `ASSOCIATION_UNRESOLVED`; Core treats that as no
detections and fails closed.
