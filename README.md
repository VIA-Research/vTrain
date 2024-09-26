# vTrain: A Simulation Framework for Evaluating Cost-effective and Compute-optimal Large Language Model Training

vTrain is a profiling-driven simulator designed to provide AI practitioners a fast yet accurate software framework to determine an efficient and cost-effective LLM training system configuration.

For more details about this work, please refer to [our paper](https://arxiv.org/abs/2312.12391) published in MICRO`24.

## Setup

### Using Docker

```
# Build Dockerfile (it can take tens of minutes)
cd ${VTRAIN_PATH}/docker
docker build -t [TAG] .

# Create docker container from the image
cd ..
docker run -it --name vtrain --gpus all -v ./:/workspace/vTrain [TAG] /bin/bash
```

In the docker container, install dependencies:
```
# Install vTrain profiling module
cd /workspace/vTrain/profiler
pip install -r requirements.txt
python setup.py install

# Install required modules for vTrain
cd ..
pip install -r requirements.txt
```

## Example

### Prediction of single-iteration time

To run vTrain simulation, you need to write a simulation configuration file.
- The example configuration is given in `config` directory, which is for simulating MT-NLG 530B training with (8, 12, 35)-way 3D parallelism using 3,360 A100 GPUs.
- For more details about the configuration parameters, please refer to the docstring of `vTrainConfig` class.

Then, load the configuration file as follows:
```
from src.predictor import vTrain
from src.config import vTrainConfig

config = vTrainConfig.load_from_file("/path/to/your/CONFIG.json")
```

Finally, instantiate a simulation object with the configuration and run the simulation by calling it:

```
sim = vTrain(config)
result, breakdown = sim()
predicted_iter_time = max(result.values())
```

`breakdown` contains the latency breakdown into compute and communication time for each device.

The full example script can be found in `example.py`.

### Visualization of execution graph

After running the simulation, you can visualize the simulated execution graph:

```
sim.show_graph()
```

## Reproducing validation results

Validation results can be simply reproduced by running vTrain as:

```
# single-node configuration example
python example.py -c config/validation/single/config_val_single_0818.json

# multi-node configuration example
python example.py -c config/validation/multi/config_val_175B_8_4_16_6.json
```

All simulation configurations used for validation can be found in `config/validation`.

## Acknowledgment

This research is funded by the generous support from the following organizations:
- National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (NRF-2021R1A2C2091753)
- Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No.RS-2024-00438851, (SW Starlab) High-performance Privacy-preserving Machine Learning System and System Software), (No.RS-2024-00402898, Simulation-based High-speed/High-Accuracy Data Center Workload/System Analysis Platform), and (No.RS-2024-00395134, DPU-Centric Datacenter Architecture for Next-Generation AI Devices)
- Samsung Advanced Institute of Technology (SAIT)
- Samsung Electronics Co., Ltd

## Citation

Jehyeon Bang, Yujeong Choi, Myeongwoo Kim, Yongdeok Kim, and Minsoo Rhu, "[vTrain: A Simulation Framework for Evaluating Cost-effective and Compute-optimal Large Language Model Training](https://arxiv.org/abs/2312.12391)," The 57th IEEE/ACM International Symposium on Microarchitecture (MICRO-57), Austin, TX, Nov. 2024
