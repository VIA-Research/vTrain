# vTrain Profiling Module

vTrain profiler collects all CUDA traces between `init_trace()` and `finish_trace()`.

Traces include 1) CUDA Runtime API calls, 2) CUDA kernels, and 3) memory operations.

There are 5 types of traces, and each trace type is formatted as follows:
- `RUNTIME`/`DRIVER`
    - `[start time],[duration (ns)],[RUNTIME | DRIVER],[cbid],[process id],[thread id],[correlation id]`
    - For cudaLaunchKernel API, the trace of the corresponding kernel has the same correlation id.
- `KERNEL`
    - `[start time],[duration (ns)],KERNEL,[kernel name],[device id],[context id],[stream id],[gridX],[gridY],[gridZ],[blockX],[blockY],[blockZ],[correlation id]`
- `MEMCPY`/`MEMSET`
    - `[start time],[duration (ns)],[MEMCPY | MEMSET],[kind],[device id],[context id],[stream id],[correlation id]`
    - For copy-/memset-kind numbers, please refer to `CUpti_ActivityMemcpyKind` and `CUpti_ActivityMemoryKind` in the CUPTI document.

To mark certain points to analyse, please use `timestamp(msg)`, which creates a timestamp with the given `msg` into traces.
