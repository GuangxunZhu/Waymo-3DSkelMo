import argparse
import os
import subprocess
from pathlib import Path

import yaml


def build_ctx(globals_cfg: dict) -> dict:
    """
    Build a formatting context from `globals` in YAML.

    Always provides:
      - {root_dir}

    For each string global key (e.g., waymo_dir: "waymo"), also provides:
      - {waymo_dir} = "waymo"
      - {waymo_dir_abs} = "/abs/path/to/{root_dir}/waymo"
    """
    root_dir = Path(globals_cfg["root_dir"]).expanduser().resolve()
    ctx = {"root_dir": str(root_dir)}

    for k, v in globals_cfg.items():
        if k == "root_dir":
            continue
        ctx[k] = v
        if isinstance(v, str):
            ctx[f"{k}_abs"] = str((root_dir / v).expanduser().resolve())

    return ctx


def format_args(arg_dict: dict, ctx: dict) -> list[str]:
    """
    Convert YAML args into a CLI list.

    Rules:
      - None: skip the argument entirely
      - bool:
          True  -> add flag only (e.g., --single-motion)
          False -> skip
      - list:
          add flag once then append multiple values
          e.g., splits: [training, validation] -> --splits training validation
      - everything else:
          --key value
    """
    out: list[str] = []
    for k, v in (arg_dict or {}).items():
        # Support keys that already include '--'
        flag = k if str(k).startswith("--") else f"--{k}"

        if v is None:
            continue

        if isinstance(v, bool):
            if v:
                out.append(flag)
            continue

        if isinstance(v, (int, float)):
            out += [flag, str(v)]
            continue

        if isinstance(v, list):
            out.append(flag)
            out += [str(item).format(**ctx) for item in v]
            continue

        out += [flag, str(v).format(**ctx)]

    return out


def resolve_project_root(config_path: Path, project_root_arg: str | None) -> Path:
    """
    Determine project root for:
      - resolving relative script paths
      - setting subprocess working directory (cwd)

    Default behavior:
      - If config is configs/pipeline.yaml, project root becomes parent of configs/ (repo root).
    """
    if project_root_arg:
        return Path(project_root_arg).expanduser().resolve()
    return config_path.parent.parent.resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline YAML")
    ap.add_argument("--project-root", default=None, help="Override project root (cwd for subprocesses)")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    project_root = resolve_project_root(config_path, args.project_root)
    ctx = build_ctx(cfg["globals"])
    steps = cfg["steps"]

    print(f"Project root: {project_root}")
    print("CTX keys:", ", ".join(sorted(ctx.keys())))

    # Make child scripts print immediately (helps tqdm / logs)
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    for i, step in enumerate(steps, 1):
        name = step.get("name", f"step_{i}")
        env_name = step["env"]

        script_rel = Path(step["script"])
        script_abs = (project_root / script_rel).resolve()
        if not script_abs.exists():
            raise FileNotFoundError(f"Script not found: {script_abs}")

        # Use --no-capture-output so tqdm/progress updates are visible
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", env_name,
            "python", "-u", str(script_abs),
        ] + format_args(step.get("args", {}), ctx)

        print(f"\n[{i}/{len(steps)}] {name}")
        print("  " + " ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=str(project_root), env=child_env)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
