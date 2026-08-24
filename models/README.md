# Required model artifacts

The model binaries are deliberately not stored in Git. An operational deployment must place the
owner-approved artifacts at the paths below. `signatus-launch --check-only` validates every SHA-256
digest before allowing the stack to start.

| Artifact | Required path | SHA-256 |
| --- | --- | --- |
| YuNet face detector | `models/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| SFace recognizer | `models/face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| YOLO OpenVINO metadata | `models/yolo26s100e18b_int8_openvino_model/metadata.yaml` | `c5300971f889d5d0741d9bd42d009086e0979c36318290e94c11ca2cf5fb25c6` |
| YOLO OpenVINO graph | `models/yolo26s100e18b_int8_openvino_model/yolo26s100e18b.xml` | `925997fab66c3c38e119523beddbfb57ea77baab0a277682435f69deb5f07772` |
| YOLO OpenVINO weights | `models/yolo26s100e18b_int8_openvino_model/yolo26s100e18b.bin` | `8e38799672cc650fde3d76678d6c365ba388479c820131de040d67cb8cb946dd` |

Copy these files through an owner-approved secure channel. Do not substitute similarly named files:
the approved class policy and face-matching behavior depend on these exact bytes.

The YOLO metadata declares the Ultralytics AGPL-3.0 license. YuNet is distributed by OpenCV Zoo
under its model-directory MIT license. Confirm the SFace weight's redistribution and training-data
terms, and the application's Ultralytics licensing obligations, before distributing a deployment
bundle outside the approved environment.
