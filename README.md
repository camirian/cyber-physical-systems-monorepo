# Coursera Machine Learning Specialization: Applied Engineering Edition

This repository contains production-grade implementations and engineering-focused adaptations of the **Coursera Machine Learning Specialization by DeepLearning.AI**.

Rather than just capturing notes, this project bridges the gap between ML theory and **Robotics, Firmware, and Systems Engineering**.

[View the Structural Roadmap](ROADMAP.md)

## 🚀 Key Differentiators (The "10x" Upgrade)

*   **Production-Grade Python:** Algorithms are refactored from notebooks into a modular Python package (`src/coursera_ml/`) with type hinting, docstrings, and clean NumPy vectorization.
*   **🤖 Robotics Bridge:** Theoretical concepts are applied to real-world robotics problems, such as **ToF Sensor Calibration** and **Motor Fault Detection**.
*   **Modern MLOps:** Integrated **GitHub Actions CI/CD** pipeline for automated testing (`pytest`), linting (`ruff`), and static analysis (`mypy`).
*   **Test-Driven Development:** Every algorithm is verified with a comprehensive test suite to ensure mathematical and structural correctness.

## 📁 Project Structure

```text
.
├── .github/workflows/      # CI/CD Automation
├── src/coursera_ml/        # Core Package (Production Algorithms)
│   ├── course_1/           # Supervised Learning
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   └── robotics_calibration.py  <-- Applied Example
│   └── course_2/           # Advanced Algorithms (Neural Networks)
├── tests/                  # Pytest Suite
├── course-1-supervised-ml/ # SQN Learning Notes
├── course-2-advanced-learning-algorithms/
├── course-3-unsupervised-learning-recommenders-rl/
├── pyproject.toml          # Project Metadata & Tooling
└── README.md
```

## 🛠️ Getting Started

Install the project in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

Run the tests to verify the implementations:

```bash
pytest tests/
```

Run the robotics calibration example:

```bash
python src/coursera_ml/course_1/robotics_calibration.py
```

## Enhanced Learning Experience

This repository goes beyond standard note-taking by incorporating interactive and visual elements to deepen understanding:

*   **📊 Visualizations:** Complex concepts like Neural Networks, Decision Trees, and K-Means clustering are visualized directly in the markdown using **Mermaid diagrams**.
*   **💻 Code Snippets:** Practical **Python/NumPy** implementations of key functions (e.g., Sigmoid, ReLU) are provided alongside the theory.
*   **navigation:** All notes include a **Table of Contents** for quick navigation to specific topics.

## Note-Taking System (SQN)

This project uses a "Structured Queryable Notes" (SQN) system to facilitate active learning and retention.

### Philosophy: From Scribe to Architect

The goal is not just to transcribe lecture content, but to actively question, structure, and connect the material to build a robust knowledge base.

### How to Use

For each week, a pre-filled note file (e.g., `week-1-notes.md`) is provided. The system uses the following legend to categorize information:

*   🔑 **Key Definition:** For critical, must-know vocabulary.
*   ❓ **Question:** For things you don't understand or want to explore later.
*   🔗 **Connection:** For links to your existing knowledge.
*   💡 **Insight:** For "aha!" moments or key takeaways.
*   ⚠️ **Warning:** For common mistakes, pitfalls, or important limitations.

At the end of each module, fill out the **Module Summary** section from memory *before* reviewing your notes to solidify your learning.

A blank template, `note_template.md`, is available in the root directory for any additional notes or future courses.
