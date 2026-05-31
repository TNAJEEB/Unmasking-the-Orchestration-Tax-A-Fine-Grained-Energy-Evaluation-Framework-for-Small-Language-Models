Copyright (c) 2026 Muhammad Talha Najeeb. All Rights Reserved.

This source code and its associated metrics are the intellectual property of the author. 
Unauthorized copying, modification, distribution, or reuse of this software for academic, 
commercial, or personal purposes is strictly prohibited without explicit written permission.

# Unmasking the Orchestration Tax: A Fine-Grained Energy Evaluation Framework for Small Language Models

Official code repository and hardware telemetry testbed implementation for the Master's Thesis submitted in partial fulfillment of the requirements for the degree of Master of Science at Aarhus University.

## Abstract
The rapid proliferation of Small Language Models (SLMs) offers a promising paradigm shift for decentralized, resource-constrained edge deployments. However, current evaluation standards (such as the foundational SLM-Bench) utilize macroscopic, system-level measurements that treat inference as an opaque black box, obscuring critical architectural bottlenecks. 

This repository introduces a fine-grained, component-level hardware telemetry framework that extends the SLM-Bench methodology. Deployed on edge-representative hardware, the testbed isolates the exact microjoule energy consumption of the CPU, GPU, and System RAM across discrete execution phases. Key findings reveal an inescapable "Orchestration Tax," where non-accelerator components account for over 30% of the active power budget during token generation, and a "De-Quantization Tax" that inflates active decoding energy by up to 59.5% for ultra-small architectures (<3B parameters) under 4-bit NormalFloat (NF4) compression.

## Experimental Environment & Hardware Topography
To ensure telemetry steady-state reproducibility, hardware dynamic scaling was disabled at the BIOS/OS level.
- **CPU:** Intel® Xeon® W-2133 Processor (6 Cores, 12 Threads, 3.60 GHz)
- **GPU:** NVIDIA RTX 5060 Ti (16GB GDDR6 VRAM, Clock locked to 2450 MHz)
- **RAM:** 128.0 GB ECC DDR4 RAM
- **OS Baseline:** Headless Ubuntu 22.04.5 LTS (`gdm3` disabled)

## Core Capabilities & Isolated Granularities
The framework systematically evaluates inference boundaries across:
1. **Component-Level (Category 1):** Real-time `pynvml` integration for GPU tracking, synced with native Linux Intel RAPL microjoule register polling for CPU/DRAM tracking.
2. **Lifecycle Steps (Category 2):** Distinct temporal isolation of tokenization setup vs. multi-gigabyte tensor weight transfers into VRAM.
3. **Operational Phases (Category 3):** Strict manual control of the neural auto-regressive loop to separate compute-bound *Prefill* paths from memory-bandwidth-bound *Decoding* loops (enforced with `torch.cuda.synchronize()`).
4. **Precision Variations (Category 4):** Comparative tracking of native 16-bit precision (`torch.float16`) against state-of-the-art 4-bit Quantization (`NF4` configuration via `bitsandbytes`).
5. **Data & Domain Granularity (Category 5):** Maps hardware performance dynamics directly as a function of scaling input and output sequence lengths across 19 evaluation datasets spanning distinct heuristic tasks.

## Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/TNAJEEB/unmasking-the-orchestration-tax.git](https://github.com/TNAJEEB/unmasking-the-orchestration-tax.git)
cd unmasking-the-orchestration-tax
```

### 2. Configure Environment
```bash
pip install -r requirements.txt
```

### 3. Run Telemetry Evaluation Pipelines
To run the evaluation in native 16-bit floating-point precision:
```bash
sudo python master_benchmark_fp16.py
```

To run the evaluation utilizing quantized 4-bit NormalFloat precision:
```bash
sudo python master_benchmark_nf4.py
```
