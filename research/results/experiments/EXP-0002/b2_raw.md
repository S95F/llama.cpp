### pass 1 arm dist0
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8187 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 8187 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |         mcmoe |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.81 GiB |    34.66 B | CUDA       |  99 |         40 |           528 |            tg64 |          9.97 ± 2.66 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.81 GiB |    34.66 B | CUDA       |  99 |         40 |          1024 |            tg64 |         14.83 ± 1.33 |

build: a250f1793 (11899)
### pass 1 arm default
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8187 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 8187 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |         mcmoe |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.81 GiB |    34.66 B | CUDA       |  99 |         40 |           528 |            tg64 |          8.18 ± 2.09 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.81 GiB |    34.66 B | CUDA       |  99 |         40 |          1024 |            tg64 |         13.16 ± 1.84 |

build: a250f1793 (11899)
### pass 2 arm default
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8187 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 8187 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |         mcmoe |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.81 GiB |    34.66 B | CUDA       |  99 |         40 |           528 |            tg64 |          8.25 ± 2.11 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.81 GiB |    34.66 B | CUDA       |  99 |         40 |          1024 |            tg64 |         13.10 ± 1.84 |

build: a250f1793 (11899)
### pass 2 arm dist0
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8187 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 8187 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |         mcmoe |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.81 GiB |    34.66 B | CUDA       |  99 |         40 |           528 |            tg64 |          8.17 ± 2.16 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.81 GiB |    34.66 B | CUDA       |  99 |         40 |          1024 |            tg64 |         13.38 ± 1.86 |

build: a250f1793 (11899)
