import os
import numpy as np

SH_C0 = 0.28209479177387814


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# Standard 3DGS PLY field order that UE5 native importer expects.
# Missing fields are padded with zeros.
_STANDARD_FIELDS = (
    ["x", "y", "z"],
    ["nx", "ny", "nz"],                            # normals — zeros if absent
    ["f_dc_0", "f_dc_1", "f_dc_2"],
    [f"f_rest_{i}" for i in range(45)],            # SH degree-3 rest bands
    ["opacity"],
    ["scale_0", "scale_1", "scale_2"],
    ["rot_0", "rot_1", "rot_2", "rot_3"],
)
_STANDARD_FIELD_LIST = [f for group in _STANDARD_FIELDS for f in group]


class PLYtoUE5Converter:
    """
    Fixes a 3DGS PLY file so UE5's native Gaussian Splat importer accepts it.

    Root cause of 'white blobs': SHARP outputs DC-only splats with no normals
    and no f_rest SH bands. UE5's importer counts fields by position — when
    nx/ny/nz and f_rest_* are absent, it misreads colour/scale/opacity and
    falls back to white.

    Fix: pad the missing fields with zeros so the file matches the full
    standard 3DGS field layout (same as a working training-pipeline output).

    UE5 native handles f_dc/scale/opacity conversions internally, so by
    default those toggles are OFF. Only turn them ON if you're using a
    community plugin that expects pre-converted values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_suffix": ("STRING", {"default": "_ue5", "multiline": False}),
                # ---- padding (main fix) ----
                "pad_to_standard_layout": ("BOOLEAN", {"default": True,
                    "tooltip": "Add missing nx/ny/nz and f_rest_* fields as zeros. "
                               "Required for UE5 native importer — this is the primary fix."}),
                # ---- optional raw-value conversions ----
                "convert_colors":   ("BOOLEAN", {"default": False,
                    "tooltip": "SH DC → sigmoid RGB. Leave OFF for UE5 native (it converts internally)."}),
                "convert_scale":    ("BOOLEAN", {"default": False,
                    "tooltip": "log(scale) → exp(scale). Leave OFF for UE5 native."}),
                "convert_opacity":  ("BOOLEAN", {"default": False,
                    "tooltip": "logit → sigmoid [0,1]. Leave OFF for UE5 native."}),
                "normalize_rotations": ("BOOLEAN", {"default": True,
                    "tooltip": "Normalise quaternions to unit length."}),
            },
            "optional": {
                "input_ply_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "C:/path/to/your_splat.ply  (or drop file above)",
                }),
            },
        }

    UPLOAD_TYPE = "file"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_ply_path",)
    FUNCTION = "convert"
    CATEGORY = "3D/Gaussian Splat"
    OUTPUT_NODE = True

    @staticmethod
    def _resolve_input_path(input_ply_path: str, ply_upload: str) -> str:
        if ply_upload:
            try:
                import folder_paths
                candidate = os.path.join(folder_paths.get_input_directory(), ply_upload)
                if os.path.isfile(candidate):
                    return candidate
            except Exception:
                pass
        path = (input_ply_path or "").strip()
        if path:
            return path
        raise ValueError("No PLY file provided — drop a file onto the node or paste a path.")

    def convert(
        self,
        output_suffix: str,
        pad_to_standard_layout: bool,
        convert_colors: bool,
        convert_scale: bool,
        convert_opacity: bool,
        normalize_rotations: bool,
        input_ply_path: str = "",
        **kwargs,
    ):
        ply_upload = kwargs.get("ply_upload", "")
        source_path = self._resolve_input_path(input_ply_path, ply_upload)

        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"PLY file not found: {source_path}")

        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            raise ImportError("plyfile is not installed. Run: pip install plyfile")

        print(f"[PLY→UE5] Reading: {source_path}")
        ply = PlyData.read(source_path)
        verts_el = ply["vertex"]
        verts = verts_el.data
        n_splats = len(verts)
        names = list(verts.dtype.names)
        data = {name: verts[name].copy() for name in names}

        print(f"[PLY→UE5] {n_splats:,} splats  |  fields: {names}")

        # ---- 1. Pad missing fields to standard layout -------------------
        if pad_to_standard_layout:
            added = []
            for field in _STANDARD_FIELD_LIST:
                if field not in data:
                    data[field] = np.zeros(n_splats, dtype=np.float32)
                    added.append(field)
            if added:
                print(f"[PLY→UE5] Padded {len(added)} missing fields with zeros: "
                      f"{added[:6]}{'…' if len(added) > 6 else ''}")
            # Reorder to standard layout, then append any extra non-standard fields
            extra = [n for n in names if n not in _STANDARD_FIELD_LIST]
            names = _STANDARD_FIELD_LIST + extra
        else:
            # Keep the original field order, just sanitise existing fields
            pass

        # ---- 2. Sanitise opacity (remove inf/nan regardless of toggle) --
        if "opacity" in data:
            op = data["opacity"]
            inf_count = int(np.sum(~np.isfinite(op)))
            if inf_count:
                op = np.nan_to_num(op, nan=0.0, posinf=10.0, neginf=-10.0)
                print(f"[PLY→UE5]   opacity: sanitised {inf_count} inf/nan values")
            if convert_opacity:
                lo, hi = float(op.min()), float(op.max())
                op = _sigmoid(op)
                print(f"[PLY→UE5]   opacity: [{lo:.4f}, {hi:.4f}] → sigmoid → [{op.min():.4f}, {op.max():.4f}]")
            data["opacity"] = op.astype(data["opacity"].dtype)

        # ---- 3. Colors (f_dc SH → sigmoid linear RGB) -------------------
        if convert_colors:
            for f in ["f_dc_0", "f_dc_1", "f_dc_2"]:
                if f in data:
                    lo, hi = float(data[f].min()), float(data[f].max())
                    data[f] = _sigmoid(data[f] * SH_C0 + 0.5).astype(data[f].dtype)
                    print(f"[PLY→UE5]   {f}: [{lo:.4f}, {hi:.4f}] → sigmoid → [{data[f].min():.4f}, {data[f].max():.4f}]")

        # ---- 4. Scale (log → exp) ---------------------------------------
        if convert_scale:
            for f in [n for n in names if n.startswith("scale_")]:
                if f in data:
                    lo, hi = float(data[f].min()), float(data[f].max())
                    data[f] = np.exp(np.clip(data[f], -10.0, 5.0)).astype(data[f].dtype)
                    print(f"[PLY→UE5]   {f}: [{lo:.4f}, {hi:.4f}] → exp → [{data[f].min():.4f}, {data[f].max():.4f}]")

        # ---- 5. Rotations (normalise to unit quaternions) ---------------
        rot_fields = sorted([n for n in names if n.startswith("rot_")])
        if normalize_rotations and len(rot_fields) == 4:
            q = np.stack([data[f].astype(np.float64) for f in rot_fields], axis=1)
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            q = q / norms
            for i, f in enumerate(rot_fields):
                data[f] = q[:, i].astype(data[f].dtype)
            print(f"[PLY→UE5]   rotations: normalised")

        # ---- Write output -----------------------------------------------
        if ply_upload:
            try:
                import folder_paths
                out_dir = folder_paths.get_output_directory()
            except Exception:
                out_dir = os.path.dirname(source_path)
        else:
            out_dir = os.path.dirname(source_path)

        base = os.path.splitext(os.path.basename(source_path))[0]
        output_ply_path = os.path.join(out_dir, f"{base}{output_suffix}.ply")

        # Build structured dtype and array in the correct field order
        dtype = np.dtype([(n, data[n].dtype) for n in names])
        out_array = np.empty(n_splats, dtype=dtype)
        for n in names:
            out_array[n] = data[n]

        el = PlyElement.describe(out_array, "vertex")
        PlyData([el], text=False).write(output_ply_path)

        size_mb = os.path.getsize(output_ply_path) / 1e6
        print(f"[PLY→UE5] Saved: {output_ply_path}  ({size_mb:.1f} MB)")
        return (output_ply_path,)


NODE_CLASS_MAPPINGS = {
    "PLYtoUE5Converter": PLYtoUE5Converter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PLYtoUE5Converter": "PLY SH \u2192 UE5 Converter",
}
