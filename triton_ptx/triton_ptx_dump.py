# Minimal, robust PTX dump for Triton kernels (jit + autotune).
# Usage: from triton_ptx_dump import dump_ptx_once

from pathlib import Path
import shutil

def _infer_kernel_name(kernel) -> str:
    name = getattr(kernel, "__name__", None)
    if not name:
        fn = getattr(kernel, "fn", None)
        name = getattr(fn, "__name__", None) if fn is not None else None
    return name or "triton_kernel"

def _iter_compiled_objs(cache):
    """
    Yield compiled objects that expose .asm (dict with 'ptx'), handling
    several cache layouts seen across Triton versions / autotune.
    """
    if not cache:
        return

    # Case A: {device_id: {signature: compiled_obj}}
    try:
        # Peek at one value to decide
        first_val = next(iter(cache.values()))
        if hasattr(first_val, "items"):  # nested dict
            for bucket in cache.values():
                for obj in bucket.values():
                    if hasattr(obj, "asm"):
                        yield obj
            return
    except StopIteration:
        return

    # Case B: {key: compiled_obj}  (compiled_obj has .asm)
    for v in cache.values():
        if hasattr(v, "asm"):
            yield v
            continue
        # Case C: {key: (compiled_obj, config)} or [compiled_obj, config]
        if isinstance(v, (list, tuple)):
            for e in v:
                if hasattr(e, "asm"):
                    yield e
                    break
        # Case D: {key: config}  -> nothing compiled stored here
        # (autotuner may stash binaries elsewhere; we skip)

def dump_ptx_once(kernel, out_dir=".", name=None, dump_all_variants=True, verbose=True):
    """
    Dump PTX for a compiled Triton kernel.
    Call this *after* the first successful launch so the cache is populated.

    Returns list of written file paths.
    """
    if not dump_all_variants and getattr(kernel, "_ptx_dumped", False):
        return []

    cache = getattr(kernel, "cache", None)
    if cache is None:
        if verbose:
            print("[ptx-dump] kernel has no 'cache' attr yet (not compiled?)")
        return []

    out = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = (name or _infer_kernel_name(kernel)).replace("/", "_")

    found_any = False
    variant_idx = 0
    for obj in _iter_compiled_objs(cache):
        found_any = True
        ptx = getattr(obj, "asm", {}).get("ptx", None)
        if not ptx:
            continue
        if dump_all_variants:
            fname = out_dir / f"{base}.v{variant_idx}.ptx"
            variant_idx += 1
        else:
            fname = out_dir / f"{base}.ptx"
            if fname.exists():
                kernel._ptx_dumped = True
                return [str(fname)]
        fname.write_text(ptx)
        out.append(str(fname))
        if verbose:
            print(f"[ptx-dump] wrote {fname}")
        if not dump_all_variants:
            kernel._ptx_dumped = True
            return out

    if not found_any and verbose:
        print("[ptx-dump] no compiled objects with .asm found in cache")

    if out and not dump_all_variants:
        kernel._ptx_dumped = True
    return out


def _collect_triton_bins(src_root: Path, dst_root: Path, clear_numbered=True):
    """
    Move (copy) compiled artifacts from hashed cache folders to
    sequential subfolders dst_root/0, dst_root/1, ... in compile order.
    """
    # Optionally clear old numbered folders
    if clear_numbered:
        for p in sorted(dst_root.glob("[0-9]*")):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

    # Find PTX files by mtime to define a stable order
    ptx_files = sorted(src_root.rglob("*.ptx"), key=lambda p: p.stat().st_mtime)
    exts = [".ptx", ".cubin", ".llir", ".ttgir", ".ttir", ".source", ".json"]

    for i, ptx in enumerate(ptx_files):
        out_sub = dst_root / f"{i}"
        out_sub.mkdir(parents=True, exist_ok=True)
        stem = ptx.stem  # e.g., 'matmul_kernel'
        # Copy the PTX and its companion files if present
        for ext in exts:
            src = ptx.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, out_sub / f"{stem}{ext}")
        print(f"[ptx-dump] wrote {out_sub}/")
