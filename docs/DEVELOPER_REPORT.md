# Developer Report

## Approved and implemented

- Linux deployment target.
- Python Core and local full-screen PySide6 GUI.
- WebSocket events from AI Service to Core.
- REST commands from Core to AI Service.
- Process-memory-only track records.
- Latest cached YOLO detection result for PPE commands.
- Fourth GUI outcome: `FACE_CAPTURE_FAILED`.
- One-second face retry cooldown.
- Three failed face captures before the track waits for exit.
- Track-lost event clears the track and permits a new screening.
- AI-owned, double-buffered shared-memory camera preview.
- Approved `single_person_frame` PPE association with a persistent GUI warning.
- Exact 0–10 model class list and Core-owned semantic PPE mapping.
- `none` total-absence override and fail-closed missing-vest handling.
- OpenCV YuNet face localization and SFace FP32 ONNX descriptor extraction.
- Core-owned cosine-similarity matching with owner-approved provisional threshold `0.35`.

## Deployment work still needed

1. USB camera model and Linux OpenCV source, usually an index such as `0` or a
   device path such as `/dev/video0`.
2. Hardware validation of OpenVINO inference, ByteTrack, shared-memory preview,
   and HDMI rendering on the deployment box.
3. Real SFace descriptors for enrolled workers and representative threshold-validation data.

## Approved PPE association

The service defaults to the owner-approved `single_person_frame` adapter. It
returns all non-person detections only when the cached frame contains exactly the
requested tracked person. Any other frame returns `ASSOCIATION_UNRESOLVED`, which
Core handles as no detections and therefore non-compliant. The operator must
physically enforce the GUI's permanent `ONE PERSON AT A TIME` warning.

## Threshold warning

The owner approved minimum cosine similarity `0.35` for functional v1 integration
with SFace. It remains provisional: run a representative false-accept/false-reject
experiment using the deployment camera, placement, and lighting before calling
the system production-stable.

## Video display boundary

Shared memory is the approved v1 local preview transport. AI Service owns capture,
Core never relays pixels, and GUI only copies the latest stable BGR frame plus
presentation-only detection overlays from shared memory. The overlays do not
participate in screening decisions. Any different transport or process-boundary
change still requires owner approval.
