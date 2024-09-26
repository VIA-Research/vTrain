import torch
import vtrain_profiler as vp

'''
vTrain profiler collects all CUDA traces between `init_trace()` and `finish_trace()`.

Traces include 1) CUDA Runtime API calls, 2) CUDA kernels, and 3) memory operations.
There are 5 types of traces, and each trace type is formatted as follows:
	- RUNTIME/DRIVER
		- [start time],[duration (ns)],[RUNTIME | DRIVER],[cbid],[process id],[thread id],[correlation id]
		- For cudaLaunchKernel API, the trace of the corresponding kernel has the same correlation id.
	- KERNEL
		- [start time],[duration (ns)],KERNEL,[kernel name],[device id],[context id],[stream id],[gridX],[gridY],[gridZ],[blockX],[blockY],[blockZ],[correlation id]
	- MEMCPY/MEMSET
		- [start time],[duration (ns)],[MEMCPY | MEMSET],[kind],[device id],[context id],[stream id],[correlation id]
		- For copy-/memset-kind numbers, please refer to `CUpti_ActivityMemcpyKind` and `CUpti_ActivityMemoryKind` in the CUPTI document.

To mark certain points to analyse, please use `timestamp(msg)`, which creates timestamp with the given `msg` into traces.
'''


vp.init_trace()

C = torch.nn.Conv2d(3, 10, 3).cuda()

for i in range(5):
	vp.timestamp(f"iter {i}")
	x = torch.randn(1, 3, 30, 30).cuda()
	out = C(x)

torch.cuda.synchronize()
vp.timestamp("done")

trace = vp.finish_trace().strip().split('\n')
trace.sort(key=lambda l: int(l.split(',')[0]))

for l in trace:
	l_split = l.split(',')

	if l_split[2] in ["RUNTIME", "DRIVER"]:
		continue

	print (l)
