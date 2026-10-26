# research/

Persistent research record for MoE expert offloading on VRAM and
PCIe constrained hardware. Start here, then STATE.md.

| file | what it is for |
|---|---|
| [STATE.md](STATE.md) | **Read first.** Restart checkpoint: status, branch, phase, exact next action |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Hardware, toolchain, model paths, and the known capability gaps |
| [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md) | Consolidated findings, each with an evidence level |
| [ROADMAP.md](ROADMAP.md) | Prioritized opportunities, reordered after every experiment |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Chronological index, dependency graph, validity conditions |
| [BENCHMARKS.md](BENCHMARKS.md) | Standard benchmark definitions and current reference numbers |
| [ARCHITECTURE.md](ARCHITECTURE.md) | llama.cpp internals the work depends on |
| [SOURCES.md](SOURCES.md) | Upstream commits and PRs, with applicability |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | Unresolved questions and what would settle each |
| [FAILED_APPROACHES.md](FAILED_APPROACHES.md) | Negative results and the conditions for revisiting them |
| `experiments/EXP-XXXX.md` | One detailed record per experiment |
| `results/` | Raw machine-readable results, never overwritten |

`moe-bench/` holds the harness and `moe-bench/RESULTS.md` the original research
log, which remains the primary record for everything before EXP-0001.

## Rules this record keeps

- Findings carry an **evidence level** (KNOWLEDGE_BASE.md); lower-confidence
  findings are never stated as facts.
- **Hit rate and uploaded bytes carry every claim**; throughput is corroboration
  and is reported with its control rows and the free host RAM.
- Raw results go to `results/`, are machine-readable, and are **never silently
  overwritten**.
- Negative results are kept with their cause and their revisit conditions.
- Correctness is a gate, not a trade-off: the cache must not change generated
  output.
