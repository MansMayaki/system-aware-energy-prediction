# Instrumentation and training reference code

This directory documents the measurement implementation used by the training pipeline.

- `codecarbon_tracker.py`: per-epoch CodeCarbon wrapper. It starts a new tracker for the monitored interval and returns CodeCarbon total energy in kWh.
- `nvml_sampler.py`: independent NVIDIA NVML sampler. It sums instantaneous power across visible GPUs and trapezoid-integrates the power trace to kWh.
- `tp_trainer_reference.py`: reference Tensor Parallel BERT trainer from the experiment code. It shows the TP sharding plan, CodeCarbon/NVML instrumentation, token accounting, and recorded configuration fields.

The complete historical training system is not required to reproduce the statistical analyses in this repository. The TP trainer is included to document the implementation path relevant to the TP measurements; it should not be interpreted as a full standalone reproduction of every BERT/GPT/FSDP training sweep.
