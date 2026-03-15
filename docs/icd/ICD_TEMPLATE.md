# Interface Control Document (ICD)

**Project / System:** [Insert System Name, e.g., Jetson Orin Edge Node to Motor Controller]
**Document Owner:** [Staff Architect]
**Date:** [Date]

## 1. System Boundaries (DSRP Alignment)
*   **Distinctions:** What this interface IS and IS NOT.
*   **Systems:** The subsystems interacting across this boundary.
*   **Relationships:** The exact physical/data linkage.
*   **Perspectives:** How this interface is viewed from the Edge Compute vs. the Hardware Actuator.

## 2. Telemetry Ingestion (Sensor -> AI)
Define the exact synthetic or physical data mapping.

| Parameter | Sensor/Source | Datatype | Range / Bounding Box | Physical Unit | Expected Frequency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `angular_vel` | IMU 6-DOF | `float32` | [-10.0, 10.0] | rad/s | 1000 Hz |
| `torque_feedback` | Motor Encoder | `float32` | [-5.0, 5.0] | N*m | 500 Hz |

*Note: All data MUST be checked for NaN and out-of-bounds physical anomalies before entering the PyTorch inference graph.*

## 3. Action Output (AI -> Hardware)
Define the exact control vectors emitted by the physics-informed model.

| Parameter | Actuator | Datatype | Safe Operating Range | Physical Unit | Rollback Default |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cmd_torque` | Momentum Wheel | `float16` | [-1.5, 1.5] | N*m | 0.0 N*m |

## 4. Blast Radius & Graceful Degradation
*   **Worst-Case Scenario:** What happens if the `cmd_torque` output flatlines at max?
*   **Automated Safety Rollback:** Define the hardcoded threshold at which the Edge Node bypasses the AI and triggers a mechanical/firmware shutdown.
*   **Verification Loop:** What is the specific "Leading Metric" that indicates the AI is diverging from the expected physical reality?

## 5. Latency Budget
*   Maximum allowable time from Sensor Read -> Inference -> Actuation: **[X] ms**
*   Target TensorRT execution time: **[Y] ms**
