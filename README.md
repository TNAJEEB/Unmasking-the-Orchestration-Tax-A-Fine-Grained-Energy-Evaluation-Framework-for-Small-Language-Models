# Unmasking-the-Orchestration-Tax-A-Fine-Grained-Energy-Evaluation-Framework-for-Small-Language-Models

## Project Description

As generative Artificial Intelligence transitions from hyper-scale cloud infrastructures to localized edge environments, optimizing the environmental and hardware sustainability of inference pipelines has become a critical bottleneck[cite: 1]. While macroscopic, software-level benchmarks aggregate power consumption into a singular, end-to-end cost metric, they treat inference as a "black box"—permanently obscuring underlying physical inefficiencies[cite: 1]. 

This repository presents a fine-grained, multi-threaded hardware telemetry testbed designed to profile Small Language Models (SLMs) across five distinct operational layers[cite: 1, 2]. Built on an isolated, headless HP Z4 G4 workstation featuring an Intel® Xeon® CPU and an NVIDIA RTX 5060 Ti GPU, this framework bypasses high-level generation wrappers to explicitly isolate component-level power draws and algorithmic boundaries[cite: 1].

### Key Framework Benchmarking Axis:
* **Resource & Physical Granularity (Category 1):** Captures real-time GPU power bindings asynchronously via NVML alongside synchronous, zero-overhead microjoule register tracking for the CPU and System RAM using Intel's Running Average Power Limit (RAPL) architecture[cite: 1].
* **Lifecycle & Execution Granularity (Category 2):** Temporally brackets model configuration, tokenizer setup, and multi-gigabyte model weight tensor transfers from local storage to VRAM to isolate one-time "cold start" penalties from recurring run costs[cite: 1, 2].
* **Operational Phase Granularity (Category 3):** Forces manual execution loops within the causal decoder architecture to decouple the compute-bound prompt processing (*Prefill Phase*) from the memory-bandwidth-bound token generation (*Decoding Phase*)[cite: 1].
* **Precision Granularity (Category 4):** Evaluates the physical impact of dynamic weight decompression by profiling native 16-bit floating-point execution (`torch.float16`) against compressed 4-bit NormalFloat (`NF4`) quantization layers[cite: 1].
* **Data & Domain Granularity (Category 5):** Evaluates hardware performance shifts as a strict function of scaling input and output sequence lengths across a deterministic, 1,000-sample subset across various heuristic tasks[cite: 1].

The framework extensively profiles a portfolio of 15 unique causal-decoder architectures (ranging from 1B to 7B parameters) across 19 evaluation datasets spanning specialized domains, reading comprehension, reasoning, and conversational dialogue[cite: 1].
