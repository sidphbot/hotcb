# cb -- Instrumentation Callbacks

`hotcb.modules.cb` provides live-loadable, hot-swappable instrumentation callbacks for PyTorch training. It runs as a kernel-routed module within hotcb, or directly via `HotController` for standalone use.

## Callback Protocol

Every callback must implement the `HotCallback` protocol:

```python
class HotCallback(Protocol):
    id: str
    def handle(self, event: str, env: Dict[str, Any]) -> None: ...
    def set_params(self, **kwargs: Any) -> None: ...
```

### Required

- **`id`** -- stable string identifier used for CLI ops and logging.
- **`handle(event, env)`** -- called at each safe point when the callback is enabled. `event` is a string like `"train_step_end"`; `env` is a dict with runtime context (`step`, `epoch`, `model`, `loss`, etc.).
- **`set_params(**kwargs)`** -- applies hot-updated parameters. Must be idempotent.

### Optional (recommended)

- **`on_attach(env)`** -- called once after the callback is loaded. Use for setup (create directories, open writers).
- **`close()`** -- called on unload. Clean up resources.

### Example

```python
class Heartbeat:
    def __init__(self, id: str, every: int = 10):
        self.id = id
        self.every = every

    def set_params(self, **kwargs):
        if "every" in kwargs:
            self.every = int(kwargs["every"])

    def handle(self, event, env):
        step = int(env.get("step", 0))
        if step % self.every == 0:
            (env.get("log") or print)(f"[{self.id}] {event} step={step}")
```

## Dynamic Loading

Callbacks can be loaded at runtime from two sources:

### Module path (`kind: module`)
```bash
hotcb cb load timing --path hotcb.modules.cb.callbacks.timing --symbol TimingCallback
```

### Python file (`kind: python_file`)
```bash
hotcb cb load feat_viz --file /tmp/feat_viz.py --symbol FeatureVizCallback --init every=50
```

The `CallbackTarget` dataclass specifies the loading source:

```python
@dataclass
class CallbackTarget:
    kind: str    # "python_file" or "module"
    path: str    # file path or module path
    symbol: str  # class name
```

`CallbackTarget` is defined in `hotcb.ops` and re-exported from `hotcb.modules.cb`.

## Standalone Mode vs hotcb Integration

### Standalone

`HotController` can run independently with its own commands file and YAML config, without requiring `HotKernel`.

```python
from hotcb.modules.cb import HotController

ctrl = HotController(
    config_path="hotcb.yaml",
    commands_path="hotcb.commands.jsonl",
)
```

### Under hotcb

When used with `HotKernel`, the `CallbackModule` wraps `HotController` and receives routed ops from the kernel. The callback module does not poll files itself -- the kernel drives all ops.

```python
# Under hotcb (kernel handles routing)
from hotcb.kernel import HotKernel

kernel = HotKernel(run_dir="runs/exp-001")
# cb module is automatically registered at kernel.modules["cb"]
```

## Source Capture

When loading from `python_file`, the kernel captures the source to `hotcb.sources/<sha256>.py`. This enables deterministic replay even if the original file changes. See [replay.md](../replay.md) for details.

## Failure Isolation

If a callback raises an exception during `handle()`, the controller can auto-disable it (configurable via `auto_disable_on_error`). The failure is logged, training continues, and the callback can be re-enabled via CLI.
