# AMAM evaluation snapshot

The `amam-eval` branch preserves the LIBERO-Mem task checker and adds a websocket evaluator. Environment reset clears `_overshot` so it cannot leak between episodes. Initial-state selection and action/state conversion are explicit in `main.py`.

Install the dependencies and assets according to the upstream instructions and `pyproject.toml`. `python main.py --help` lists the websocket evaluator arguments. AMAM's VLM orchestration and evaluation aggregation live in AMAM separately.

The standard AMAM protocol uses seed 7, 100 initial states per task and replan16 without action striding. The separate stride2 experiment must not be mixed with this protocol. Run the full task-specific environment budget to measure overshoot. Strict SR is kept separate from the additional relaxed SR (`success OR overshot`) and completion statistics computed by the AMAM aggregator.

`amam_environment.json` records the frozen adaptations. No model weights or private runtime artifacts are included.
