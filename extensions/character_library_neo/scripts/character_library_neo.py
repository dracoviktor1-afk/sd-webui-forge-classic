import os
import json
import time
import uuid
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from modules import script_callbacks, shared, scripts, ui, processing
from modules.paths import data_path
from modules import script_callbacks

EXTENSION_DIR = os.path.dirname(__file__)
EXTENSION_CANONICAL_NAME = "character_library_neo"
CHAR_ROOT = os.path.join(data_path, "character_library_neo")
CHAR_INDEX = os.path.join(CHAR_ROOT, "characters.json")
TRAIN_SCRIPT = os.path.join(EXTENSION_DIR, "..", "train_lora.py")  # optional training script

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _read_json(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: str, payload):
    _safe_mkdir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _init_store():
    _safe_mkdir(CHAR_ROOT)
    if not os.path.exists(CHAR_INDEX):
        _write_json(CHAR_INDEX, {"characters": []})

# Character metadata now includes folder_name (the actual directory on disk)
@dataclass
class Character:
    id: str
    name: str
    folder_name: str               # actual directory name under CHAR_ROOT
    created_at: str
    updated_at: str
    identity_ref: Optional[str]
    style_refs: List[str]
    notes: str
    defaults: Dict[str, Any]
    preview: Optional[str]

def _list_characters() -> List[Character]:
    _init_store()
    idx = _read_json(CHAR_INDEX, {"characters": []})
    res: List[Character] = []
    for c in idx.get("characters", []):
        res.append(
            Character(
                id=c["id"],
                name=c.get("name", c["id"]),
                folder_name=c.get("folder_name", f"{c.get('name',c.get('id'))} -- {c['id']}"),
                created_at=c.get("created_at", ""),
                updated_at=c.get("updated_at", ""),
                identity_ref=c.get("identity_ref"),
                style_refs=c.get("style_refs", []) or [],
                notes=c.get("notes", "") or "",
                defaults=c.get(
                    "defaults",
                    {
                        "enable_identity": True,
                        "enable_style": True,
                        "identity_strength": 0.85,
                        "style_strength": 0.6,
                        "start_percent": 0.0,
                        "end_percent": 1.0,
                    },
                ),
                preview=c.get("preview"),
            )
        )
    return res

def _save_character(updated: Character):
    chars = _list_characters()
    idx = {c.id: c for c in chars}
    idx[updated.id] = updated
    payload = {
        "characters": [
            {
                "id": c.id,
                "name": c.name,
                "folder_name": c.folder_name,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "identity_ref": c.identity_ref,
                "style_refs": c.style_refs,
                "notes": c.notes,
                "defaults": c.defaults,
                "preview": c.preview,
            }
            for c in idx.values()
        ]
    }
    payload["characters"].sort(key=lambda x: (x.get("updated_at", ""), x.get("name", "")), reverse=True)
    _write_json(CHAR_INDEX, payload)

def _delete_character(char_id: str):
    cid = _normalize_char_id(char_id)
    if not cid:
        return
    # remove entry from JSON and delete matching folder_name on disk if present
    chars = _list_characters()
    kept = [c for c in chars if c.id != cid]
    _write_json(
        CHAR_INDEX,
        {
            "characters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "folder_name": c.folder_name,
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                    "identity_ref": c.identity_ref,
                    "style_refs": c.style_refs,
                    "notes": c.notes,
                    "defaults": c.defaults,
                    "preview": c.preview,
                }
                for c in kept
            ]
        },
    )
    # remove the character directory (folder_name) if it exists
    # find the folder_name for the char we deleted (from original list)
    orig = next((c for c in chars if c.id == cid), None)
    if orig:
        folder = os.path.join(CHAR_ROOT, orig.folder_name)
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)

# robust normalization for character id values coming from Gradio
def _normalize_char_id(char_id):
    # unwrap nested lists/tuples (e.g. [[id]] or [id])
    while isinstance(char_id, (list, tuple)):
        if len(char_id) == 0:
            return None
        char_id = char_id[0]

    # some Gradio versions may wrap values in dicts; try common keys
    if isinstance(char_id, dict):
        for k in ("value", "id", "name"):
            if k in char_id:
                return _normalize_char_id(char_id[k])
        # if dict is like {"0": "<id>"} pick first value
        vals = list(char_id.values())
        if vals:
            return _normalize_char_id(vals[0])
        return None

    if char_id is None:
        return None

    # final: convert to string (IDs in our store are strings)
    try:
        return str(char_id)
    except Exception:
        return None

def _char_dir(char_id: str) -> str:
    cid = _normalize_char_id(char_id)
    if not cid:
        # return root characters directory for non-selected id (caller should check for None)
        return CHAR_ROOT
    return os.path.join(CHAR_ROOT, str(cid))

def _refs_dir(char_id: str) -> str:
    cid = _normalize_char_id(char_id)
    if not cid:
        return os.path.join(CHAR_ROOT, "no_char_selected_refs")
    return os.path.join(_char_dir(cid), "refs")

def _ref_abs_path(char_id: str, ref_rel: str) -> str:
    cid = _normalize_char_id(char_id)
    if not cid or not ref_rel:
        return ""
    return os.path.join(_char_dir(cid), ref_rel).replace("\\", "/")

def _list_refs(char_id: str) -> List[str]:
    cid = _normalize_char_id(char_id)
    if not cid:
        return []
    rdir = _refs_dir(cid)
    if not os.path.isdir(rdir):
        return []
    out = []
    for fn in sorted(os.listdir(rdir)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            out.append(os.path.join("refs", fn))
    return out

def _create_character(name: str) -> Tuple[str, str]:
    _init_store()
    char_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    created = _now_iso()
    c = Character(
        id=char_id,
        name=(name or "Unnamed").strip(),
        created_at=created,
        updated_at=created,
        identity_ref=None,
        style_refs=[],
        notes="",
        defaults={
            "enable_identity": True,
            "enable_style": True,
            "identity_strength": 0.85,
            "style_strength": 0.6,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
        preview=None,
    )
    _safe_mkdir(_refs_dir(char_id))
    _save_character(c)
    return char_id, f"Created character: {c.name} ({c.id})"

def _add_refs(char_id: str, files: List[Any]) -> str:
    # allow gradio dropdown to pass list/tuple/dict
    cid = _normalize_char_id(char_id)
    if not cid:
        return "Select a character first."
    if not files:
        return "No files provided."

    rdir = _refs_dir(cid)
    _safe_mkdir(rdir)

    added = 0
    for f in files:
        src = None
        # Gradio file inputs commonly come as dicts with 'name' or as TemporaryUploadedFile with .name
        if isinstance(f, str):
            src = f
        elif hasattr(f, "name"):
            src = f.name
        elif isinstance(f, dict):
            # Some gradio versions produce {"name": "...", "tmp_path": "..."}
            # prefer "name", then any path-like values
            if "name" in f and isinstance(f["name"], str):
                src = f["name"]
            elif "tmp_path" in f and isinstance(f["tmp_path"], str):
                src = f["tmp_path"]
            else:
                # fallback: try first string value
                for v in f.values():
                    if isinstance(v, str) and os.path.exists(v):
                        src = v
                        break

        if not src or not os.path.exists(src):
            # skip invalid entries
            continue

        ext = os.path.splitext(src)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            continue

        dst = os.path.join(rdir, f"{uuid.uuid4().hex}{ext}")
        try:
            shutil.copy(src, dst)
            added += 1
        except Exception as exc:
            print(f"[character_library_neo] failed copying {src} -> {dst}: {exc}")
            continue

    if added == 0:
        return "No valid images were added."
    # update character updated_at and save
    chars = {c.id: c for c in _list_characters()}
    c = chars.get(cid)
    if c:
        c.updated_at = _now_iso()
        _save_character(c)
    return f"Added {added} reference image(s)."

def _make_preview(char_id: str) -> Optional[str]:
    cid = _normalize_char_id(char_id)
    if not cid:
        return None
    # Simple preview: pick identity_ref or first style ref and return absolute path.
    c = next((c for c in _list_characters() if c.id == cid), None)
    if not c:
        return None
    candidate = c.identity_ref or (c.style_refs[0] if c.style_refs else None)
    if not candidate:
        return None
    return _ref_abs_path(cid, candidate)

def _ui_refresh_character_list() -> Tuple[List[Tuple[str,str]], str]:
    chars = _list_characters()
    choices = [(f"{c.name} — {c.id}", c.id) for c in chars]
    msg = f"{len(choices)} character(s) loaded from: {CHAR_ROOT}"
    return choices, msg

def _ui_load_character(char_id: str):
    cid = _normalize_char_id(char_id)
    if not cid:
        return (
            [],
            [],
            None,
            [],
            "",
            True,
            True,
            0.85,
            0.6,
            0.0,
            1.0,
            None,
        )

    c = next((cc for cc in _list_characters() if cc.id == cid), None)
    if not c:
        return (
            [],
            [],
            None,
            [],
            "",
            True,
            True,
            0.85,
            0.6,
            0.0,
            1.0,
            None,
        )

    refs = _list_refs(cid)
    gallery_items = [(_ref_abs_path(cid, r), os.path.basename(r)) for r in refs]
    identity_choices = [(os.path.basename(r), r) for r in refs]
    style_choices = identity_choices

    return (
        gallery_items,
        identity_choices,
        c.identity_ref,
        c.style_refs,
        c.notes or "",
        bool(c.defaults.get("enable_identity", True)),
        bool(c.defaults.get("enable_style", True)),
        float(c.defaults.get("identity_strength", 0.85)),
        float(c.defaults.get("style_strength", 0.6)),
        float(c.defaults.get("start_percent", 0.0)),
        float(c.defaults.get("end_percent", 1.0)),
        c.preview,
    )

def _ui_save_character(
    char_id: str,
    identity_ref: Optional[str],
    style_refs: List[str],
    notes: str,
    enable_identity: bool,
    enable_style: bool,
    identity_strength: float,
    style_strength: float,
    start_percent: float,
    end_percent: float,
) -> str:
    cid = _normalize_char_id(char_id)
    if not cid:
        return "No character selected."

    c = next((cc for cc in _list_characters() if cc.id == cid), None)
    if not c:
        return "No character selected."

    refs = set(_list_refs(cid))
    identity_ref = identity_ref if identity_ref in refs else None
    style_refs = [r for r in (style_refs or []) if r in refs]

    c.identity_ref = identity_ref
    c.style_refs = style_refs
    c.notes = notes or ""
    c.updated_at = _now_iso()
    c.defaults = dict(
        enable_identity=bool(enable_identity),
        enable_style=bool(enable_style),
        identity_strength=float(identity_strength),
        style_strength=float(style_strength),
        start_percent=float(start_percent),
        end_percent=float(end_percent),
    )
    c.preview = _make_preview(cid)
    _save_character(c)
    return "Saved."

def _ui_delete_character(char_id: str):
    cid = _normalize_char_id(char_id)
    if not cid:
        return [], [], None, "No character selected."
    _delete_character(cid)
    choices, msg = _ui_refresh_character_list()
    return choices, msg

# Simple helper to spawn training (optional)
def _spawn_train_job(char_id: str, steps: int = 200, lr: float = 1e-4) -> str:
    if not os.path.exists(TRAIN_SCRIPT):
        return "Training script not found. Install or place train_lora.py in the extension folder to enable training."
    args = [
        "python",
        TRAIN_SCRIPT,
        "--char-id", char_id,
        "--data-dir", _char_dir(char_id),
        "--output-dir", os.path.join(_char_dir(char_id), "trained"),
        "--steps", str(steps),
        "--lr", str(lr),
    ]
    try:
        # spawn background process (non-blocking)
        subprocess.Popen(args, cwd=EXTENSION_DIR)
        return "Training started in background (check terminal)."
    except Exception as e:
        return f"Failed to start training: {e}"

# ----- UI and integration hooks -----

def on_ui_tabs():
    _init_store()

    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("## Character Library (Neo)\n"
                    "Create a character from multiple reference images. Use the 'Apply' control in generation UI\n"
                    "to bind a character to your next generation. You can optionally train a small adapter (LoRA) for\n"
                    "strong, LoRA-like consistency — see 'Train Adapter'.")
        with gr.Row():
            refresh_btn = gr.Button("Refresh list", variant="secondary")
            status = gr.Markdown("")

        with gr.Row():
            character_dd = gr.Dropdown(label="Character", choices=[], value=None, interactive=True)
            new_name = gr.Textbox(label="New character name", placeholder="e.g., Ayla", scale=2)
            create_btn = gr.Button("Create", variant="primary")

        with gr.Row():
            delete_btn = gr.Button("Delete character", variant="stop")
            save_btn = gr.Button("Save settings", variant="primary")
            train_btn = gr.Button("Train Adapter (LoRA)", variant="primary")
        with gr.Row():
            upload = gr.File(label="Add reference images", file_count="multiple", file_types=["image"])
            add_btn = gr.Button("Add images", variant="secondary")

        gallery = gr.Gallery(label="Reference images", columns=6, height=240, preview=True, object_fit="contain")
        preview_img = gr.Image(label="Preview image", type="filepath")

        with gr.Row():
            identity_ref_dd = gr.Dropdown(label="Identity image (InstantID)", choices=[], value=None, interactive=True)
            style_refs_dd = gr.Dropdown(label="Style/Outfit refs (IP-Adapter)", choices=[], value=[], multiselect=True, interactive=True)

        notes = gr.Textbox(label="Notes", lines=3, placeholder="Optional notes about the character…")

        with gr.Row():
            enable_identity = gr.Checkbox(label="Enable Identity Lock (InstantID)", value=True)
            enable_style = gr.Checkbox(label="Enable Style Lock (IP-Adapter)", value=True)

        with gr.Row():
            identity_strength = gr.Slider(label="Identity strength", minimum=0.0, maximum=2.0, step=0.01, value=0.85)
            style_strength = gr.Slider(label="Style strength", minimum=0.0, maximum=2.0, step=0.01, value=0.6)

        with gr.Row():
            start_percent = gr.Slider(label="Start %", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
            end_percent = gr.Slider(label="End %", minimum=0.0, maximum=1.0, step=0.01, value=1.0)

        log_out = gr.Textbox(label="Status output", lines=2, interactive=False)

        # Load / refresh handlers
        def _refresh():
            choices, msg = _ui_refresh_character_list()
            return choices, msg

        ui.load(fn=_refresh, inputs=[], outputs=[character_dd, status], show_progress=False)
        refresh_btn.click(fn=_refresh, inputs=[], outputs=[character_dd, status], show_progress=False)

        create_btn.click(
            fn=lambda name: _create_character(name),
            inputs=[new_name],
            outputs=[character_dd, status],
            show_progress=True,
        ).then(fn=_refresh, inputs=[], outputs=[character_dd, status], show_progress=False)

        character_dd.change(
            fn=_ui_load_character,
            inputs=[character_dd],
            outputs=[
                gallery,
                identity_ref_dd,
                preview_img,
                style_refs_dd,
                notes,
                enable_identity,
                enable_style,
                identity_strength,
                style_strength,
                start_percent,
                end_percent,
                preview_img,
            ],
            show_progress=False,
        )

        add_btn.click(
            fn=_add_refs,
            inputs=[character_dd, upload],
            outputs=[log_out],
            show_progress=True,
        ).then(
            fn=_ui_load_character,
            inputs=[character_dd],
            outputs=[
                gallery,
                identity_ref_dd,
                preview_img,
                style_refs_dd,
                notes,
                enable_identity,
                enable_style,
                identity_strength,
                style_strength,
                start_percent,
                end_percent,
                preview_img,
            ],
            show_progress=False,
        )

        save_btn.click(
            fn=_ui_save_character,
            inputs=[
                character_dd,
                identity_ref_dd,
                style_refs_dd,
                notes,
                enable_identity,
                enable_style,
                identity_strength,
                style_strength,
                start_percent,
                end_percent,
            ],
            outputs=[log_out],
            show_progress=False,
        )

        delete_btn.click(
            fn=lambda cid: _ui_delete_character(cid),
            inputs=[character_dd],
            outputs=[character_dd, status],
            show_progress=True,
        )

        train_btn.click(
            fn=lambda cid: _spawn_train_job(cid),
            inputs=[character_dd],
            outputs=[log_out],
            show_progress=False,
        )

    return [(ui, "Characters", "characters")]

script_callbacks.on_ui_tabs(on_ui_tabs)

# -------------------------
# Add small controls into the generation UI area so users can choose "Use Character"
# We'll register a Script that returns UI components for insertion into txt2img/img2img panels.
# This approach mirrors other Forge builtin extensions which return small UI controls and
# provides a hook process_before_every_sampling where we inject preprocessors.
# -------------------------

from modules import scripts as scripts_mod

class CharacterBindingScript(scripts_mod.Script):
    sorting_priority = 50

    def title(self):
        return "Character Library Binding"

    def ui(self, is_img2img):
        with gr.Row():
            use_character = gr.Dropdown(label="Use Character", choices=[(f"{c.name} — {c.id}", c.id) for c in _list_characters()], value=None)
            apply_button = gr.Button(value="Apply Character", variant="primary")
            lock_toggle = gr.Checkbox(label="Sticky (keep across generations)", value=False)
        # These comps will be attached into the generation UI automatically by Forge
        return use_character, apply_button, lock_toggle

    def process_before_every_sampling(self, p, use_character, apply_button, lock_toggle, **kwargs):
        # Called before sampling. We'll attach IP-Adapter / InstantID conditioning if possible.
        char_id = use_character
        if not char_id:
            return

        c = next((cc for cc in _list_characters() if cc.id == char_id), None)
        if not c:
            return

        # Attempt to apply InstantID/IP-Adapter conditioning by constructing conditioning dicts
        try:
            # 1) Collect absolute paths to images
            refs_abs = [ _ref_abs_path(char_id, r) for r in (c.style_refs or [])]
            identity_abs = _ref_abs_path(char_id, c.identity_ref) if c.identity_ref else None

            # 2) Add them to process.extra_generation_params so built-in preprocessors can see them.
            # Many forge preprocessors look for fields like 'reference_images' or units; add both.
            # This is a conservative approach: we don't mutate internal UNET structures directly.
            egp = getattr(p, "extra_generation_params", {})
            egp["character_library_ref_images"] = refs_abs
            egp["character_library_identity_image"] = identity_abs
            egp["character_library_identity_strength"] = float(c.defaults.get("identity_strength", 0.85))
            egp["character_library_style_strength"] = float(c.defaults.get("style_strength", 0.6))
            egp["character_library_enable_identity"] = bool(c.defaults.get("enable_identity", True))
            egp["character_library_enable_style"] = bool(c.defaults.get("enable_style", True))
            p.extra_generation_params = egp

            # Many forge-preprocessors use presence of these keys to pick them up.
            # If you later enable the "auto-inject" path (LoRA training + auto-load), we'll replace this.
        except Exception as e:
            print(f"[character_library_neo] failed to attach character refs: {e}")

# Robust, idempotent registration for CharacterBindingScript
# This block removes any previous faulty entries and registers the script
# in the descriptor format expected by this fork's modules.scripts.initialize_scripts.

from types import SimpleNamespace

def _cleanup_existing_registration():
    # remove older fallback attributes if present
    for attr in ("character_library_neo_fallback_script", "character_library_neo_fallback", "character_library_neo_descriptor"):
        if hasattr(scripts_mod, attr):
            try:
                delattr(scripts_mod, attr)
                print(f"[character_library_neo] removed previous attribute on modules.scripts: {attr}")
            except Exception:
                pass

    # clean scripts_data list entries referencing this module or class
    if hasattr(scripts_mod, "scripts_data") and isinstance(scripts_mod.scripts_data, list):
        before = len(scripts_mod.scripts_data)
        scripts_mod.scripts_data[:] = [
            sd for sd in scripts_mod.scripts_data
            if not (
                getattr(sd, "module", None) == __name__
                or getattr(sd, "script_class", None) in (CharacterBindingScript, CharacterBindingScript.__name__)
            )
        ]
        after = len(scripts_mod.scripts_data)
        if before != after:
            print(f"[character_library_neo] cleaned {before-after} entries from modules.scripts.scripts_data")

    # clean scripts_list entries referencing this module/class/instance
    if hasattr(scripts_mod, "scripts_list") and isinstance(scripts_mod.scripts_list, list):
        before = len(scripts_mod.scripts_list)
        new_list = []
        for item in scripts_mod.scripts_list:
            try:
                # item may be class, instance, or descriptor — filter anything referencing our class/module
                if item in (CharacterBindingScript,):
                    continue
                # if it's an instance of our class, skip it
                if isinstance(item, CharacterBindingScript):
                    continue
                # sometimes entries are descriptors in legacy shapes, skip module match
                if getattr(item, "module", None) == __name__:
                    continue
                new_list.append(item)
            except Exception:
                new_list.append(item)
        scripts_mod.scripts_list[:] = new_list
        after = len(scripts_mod.scripts_list)
        if before != after:
            print(f"[character_library_neo] cleaned {before-after} entries from modules.scripts.scripts_list")

_cleanup_existing_registration()

_registered = False

# Prepare descriptor object with expected attributes (module, script_class)
descriptor = SimpleNamespace(module=__name__, script_class=CharacterBindingScript)

# Preferred path: append to scripts_data if present
try:
    if hasattr(scripts_mod, "scripts_data") and isinstance(scripts_mod.scripts_data, list):
        scripts_mod.scripts_data.append(descriptor)
        _registered = True
        print("[character_library_neo] registered script descriptor in modules.scripts.scripts_data")
except Exception as e:
    print(f"[character_library_neo] failed to append to scripts_data: {e}")

# Fallback: append the class to scripts_list if scripts_data doesn't exist
try:
    if not _registered and hasattr(scripts_mod, "scripts_list") and isinstance(scripts_mod.scripts_list, list):
        # some forks expect classes in scripts_list, so append the class (not an instance)
        scripts_mod.scripts_list.append(CharacterBindingScript)
        _registered = True
        print("[character_library_neo] appended CharacterBindingScript class to modules.scripts.scripts_list")
except Exception as e:
    print(f"[character_library_neo] failed to append to scripts_list: {e}")

# Last resort: attach descriptor as attribute so loader can pick it up later
if not _registered:
    try:
        setattr(scripts_mod, "character_library_neo_descriptor", descriptor)
        print("[character_library_neo] installed fallback descriptor on modules.scripts as 'character_library_neo_descriptor'")
        _registered = True
    except Exception as e:
        print(f"[character_library_neo] final fallback registration failed: {e}")

if _registered:
    print("[character_library_neo] registration completed successfully")
else:
    print("[character_library_neo] registration failed; your webui variant may use a different registration API.")
