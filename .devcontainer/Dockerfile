# Context: High-Fidelity Scientific Python 3.10 Sandbox
# Reason: Enforces absolute mathematical parity across distributed matrices.
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: Core extraction tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
