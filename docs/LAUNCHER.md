# Signatus CLI Supervisor

`signatus-launch` performs static deployment checks, starts the approved three
Signatus processes in order, and supervises their operational readiness.

```text
signatus-launch
  -> AI Service ready (models/components loaded; camera STOPPED)
  -> Core ready (AI event WebSocket connected)
  -> GUI running (operator controls camera through Core)
```

The launcher does not perform identity matching, PPE evaluation, authorization,
or camera capture. Those responsibilities remain in Core and the AI Service.
It does not automatically restart Core because restarting Core intentionally
clears all in-memory track records.

## Commands

Run a static preflight without starting processes or opening the camera:

```bash
signatus-launch --check-only
```

Start the full-screen deployment stack:

```bash
signatus-launch
```

Start with a development GUI window:

```bash
signatus-launch --windowed
```

Run and supervise only the AI Service and Core:

```bash
signatus-launch --no-gui
```

Use a particular deployment environment:

```bash
signatus-launch --env-file /path/to/deployment/.env
```

The env file's directory is the working directory for all child processes, so
relative model and worksite paths resolve consistently. Values already exported
in the launcher's process environment take precedence over the dotenv file.

Additional options:

- `--startup-timeout SECONDS` controls each service readiness deadline. The
  default is 60 seconds.
- `--shutdown-timeout SECONDS` controls each process's graceful stop deadline
  at each escalation stage. Shutdown requests `SIGINT`, then `SIGTERM`, and
  uses `SIGKILL` only as the final fallback. The default is 10 seconds.
- `--log-dir PATH` changes the rotating log location. The default is
  `.signatus/logs` beside the selected env file.

## Preflight boundary

Preflight is static and does not open the camera or load the inference models.
It rejects an operational launch when, among other fatal conditions:

- tracking is disabled;
- the exact approved person class, `single_person_frame` association, or
  functional `0.35` threshold has been changed;
- required model, face-model, configuration, port, or IPC infrastructure is
  absent or invalid;
- the OpenVINO metadata does not contain the exact approved class ID/name map;
- a service port or shared-memory preview name is already occupied; or
- a GUI launch has no graphical display or shared-memory preview configuration.

Missing camera hardware is a warning, not a startup failure. Invalid worker
descriptors and individual Wo.No. policies are `DATA ERROR`s: Core disables the
smallest affected scope, and `--check-only` returns `PASS WITH DATA ERRORS` if
the stack can still start. Strict 128-value finite, nonzero SFace validation is
unchanged; data is never padded, truncated, or repaired.

During startup, the AI Service initializes YOLO, YuNet, SFace, and preview IPC,
and validates the runtime YOLO class map. It does not open the camera or require
an inferred/published frame. Core then establishes its AI event connection.
Service health and camera state remain independent: both Camera `STOPPED` and
Camera `ERROR` can coexist with AI `READY`. The launcher continuously monitors
service readiness and shuts the full stack down after repeated loss.

## Shutdown and logs

Closing the GUI normally, pressing `Ctrl+C`, or sending `SIGTERM` causes an
ordered GUI, Core, then AI Service shutdown. If AI, Core, or the GUI exits
unexpectedly, the launcher reports failure and stops the remaining processes.
One per-user runtime lease prevents concurrent launchers from contending for
ports, camera, and shared memory. Child processes run in isolated process groups
so descendants are included in escalation and cleanup.

Console output is prefixed with `AI`, `Core`, `GUI`, or `launcher`. Each run has
a unique launch ID propagated to child processes. A combined 5 MiB rotating log
with three backups records timestamps, launch ID, component, PID, exit code,
and validation severity in `signatus.log`.
