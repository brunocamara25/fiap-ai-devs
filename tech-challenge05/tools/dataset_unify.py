import argparse, os, re, glob, json, random, difflib
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET
from pathlib import Path
from PIL import Image
import yaml

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# -------------------- utils --------------------
def read_yaml_file(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_text(p, lines):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def list_images_from_splits(dataset_root):
    tr = Path(dataset_root, "splits", "train.txt")
    va = Path(dataset_root, "splits", "val.txt")
    train = [ln.strip() for ln in tr.read_text().splitlines()] if tr.exists() else []
    val   = [ln.strip() for ln in va.read_text().splitlines()] if va.exists() else []
    return train, val

def scan_images_from_labels(dataset_root):
    labels_dir = Path(dataset_root, "labels")
    images_dir = Path(dataset_root, "images")
    out = []
    if not labels_dir.exists():
        return out
    for lab in labels_dir.glob("*.txt"):
        base = lab.stem
        img = None
        for ext in IMG_EXTS:
            p = images_dir / f"{base}{ext}"
            if p.exists():
                img = str(p.resolve())
                break
        if img:
            out.append(img)
    return out

def load_names_from_dataset(dataset_root):
    dy = Path(dataset_root, "data.yaml")
    if dy.exists():
        data = read_yaml_file(dy)
        names = data.get("names")
        if isinstance(names, dict):
            names = [v for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        return names
    return None

def build_id_remap(old_names, canonical_names):
    if old_names == canonical_names:
        return {i:i for i in range(len(old_names))}
    old_set, new_set = set(old_names), set(canonical_names)
    if old_set != new_set:
        return None
    name2new = {n:i for i,n in enumerate(canonical_names)}
    return {i: name2new[n] for i,n in enumerate(old_names)}

def remap_label_file(src_label, dst_label, id_map):
    lines_out = []
    with open(src_label, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            try:
                old_id = int(parts[0])
                if old_id in id_map:
                    parts[0] = str(id_map[old_id])
                    lines_out.append(" ".join(parts))
            except Exception:
                continue
    Path(dst_label).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label, "w", encoding="utf-8") as g:
        g.write("\n".join(lines_out))

def copy_or_link_image(src_img, dst_img, copy=True, resize_long=0):
    Path(dst_img).parent.mkdir(parents=True, exist_ok=True)
    if resize_long and resize_long > 0:
        try:
            with Image.open(src_img) as im:
                W,H = im.size
                if max(W,H) > resize_long:
                    scale = resize_long / float(max(W,H))
                    im = im.resize((int(W*scale), int(H*scale)), Image.BICUBIC)
                im.save(dst_img, quality=95)
            return
        except Exception:
            pass
    if copy:
        from shutil import copy2
        copy2(src_img, dst_img)
    else:
        try:
            os.link(src_img, dst_img)
        except Exception:
            from shutil import copy2
            copy2(src_img, dst_img)

def safe_basename(path):
    b = Path(path).stem
    b = re.sub(r"[^a-zA-Z0-9._-]+", "_", b)
    return b

def ensure_dir(d): Path(d).mkdir(parents=True, exist_ok=True)

def read_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(obj, p):
    ensure_dir(Path(p).parent)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def write_txt(p, lines):
    ensure_dir(Path(p).parent)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def clean_norm(x: str) -> str:
    x = (x or "").strip().lower()
    x = re.sub(r"[^\w\s/+-]+", "_", x)
    x = x.replace("-", "_").replace(" ", "_")
    x = re.sub(r"__+", "_", x)
    return x

def load_names(path):
    path = str(path)
    if path.lower().endswith((".yml", ".yaml")):
        data = read_yaml(path)
        return [str(x).strip() for x in data["names"]]
    return [ln.strip() for ln in open(path, "r", encoding="utf-8") if ln.strip()]

def cfg_get(cfg, dotted, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# --------------- LLM helpers (Ollama) ---------------
def classify_ollama_batch(label_names, classes, model="llama3", temperature=0.0, batch=50):
    try:
        import ollama
    except Exception:
        return {}
    results = {}
    classes_line = ", ".join(classes)

    for i in range(0, len(label_names), max(1, batch)):
        chunk = label_names[i:i+batch]
        prompt = (
            "Classify each of the following labels into ONE of the existing categories (do NOT create new categories):\n"
            f"Categories: {classes_line}\n"
            "Respond with one per line in the exact format: <label>: <category>\n"
            "Only use the categories provided above. Do not invent or suggest new categories.\n"
            "If a label does not fit any category, respond with '<label>: not_found'.\n\n"
            "Labels:\n" + "\n".join([f"{x}:" for x in chunk])
        )
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": float(temperature)}
            )
            text = resp["message"]["content"]
        except Exception:
            text = ""
        for line in text.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k and v:
                    results[k] = v
    return results

# --------------------- parsers ----------------------
def find_image_for_xml(xml_path, img_roots=None):
    img_roots = img_roots or []
    try:
        root = ET.parse(xml_path).getroot()
        path_tag = root.findtext("path")
        fname_tag = root.findtext("filename")
    except Exception:
        return None
    xml_dir = Path(xml_path).parent
    if path_tag:
        p = Path(path_tag)
        if not p.is_absolute():
            p = xml_dir / p
        if p.is_file():
            return str(p.resolve())
    base = Path(fname_tag).stem if fname_tag else Path(xml_path).stem
    for e in IMG_EXTS:
        p = xml_dir / f"{base}{e}"
        if p.is_file():
            return str(p.resolve())
    for sub in ("images", "image", "img", "JPEGImages", "imgs"):
        for e in IMG_EXTS:
            p = xml_dir / sub / f"{base}{e}"
            if p.is_file():
                return str(p.resolve())
    up = xml_dir.parent
    for e in IMG_EXTS:
        p = up / f"{base}{e}"
        if p.is_file():
            return str(p.resolve())
    for r in img_roots:
        r = Path(r)
        for p in r.rglob(f"{base}*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                return str(p.resolve())
    return None

def parse_boxes_from_xml(root):
    out = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            out.append((name, [x1, y1, x2, y2]))
        except Exception:
            pass
    return out

# ------------- mapeamento para canônicas -------------
def build_mapper(real_names, canon_names, synonyms, class_map, heur_token2class,
                 use_llm=False, llm_model="llama3", llm_temp=0.0, llm_batch=50,
                 auto_fuzzy=False, fuzzy_cutoff=0.92):
    canonical   = set(canon_names)
    canon_norm  = {clean_norm(n): n for n in canon_names}
    syn_norm    = {clean_norm(k): v for k, v in (synonyms or {}).items()}

    real2canon  = {}
    mapped_exact = mapped_norm = mapped_llm = mapped_heur = mapped_fuzzy = 0

    for rn in real_names:
        # 1) class_map (chave EXATA)
        if rn in class_map:
            if class_map[rn] == "ignore":
                real2canon[rn] = None
                continue
            if class_map[rn] in canonical:
                real2canon[rn] = class_map[rn]
                mapped_exact += 1
                continue

        # 2) já é canônica?
        if rn in canonical:
            real2canon[rn] = rn
            mapped_exact += 1
            continue

        rn_n = clean_norm(rn)

        # 3) class_map (chave NORMALIZADA)
        if rn_n in class_map:
            if class_map[rn_n] == "ignore":
                real2canon[rn] = None
                continue
            if class_map[rn_n] in canonical:
                real2canon[rn] = class_map[rn_n]
                mapped_norm += 1
                continue

        # 4) sinônimo direto (normalizado)
        syn_target = syn_norm.get(rn_n)
        if syn_target in canonical:
            real2canon[rn] = syn_target
            mapped_norm += 1
            continue

        # 5) nome normalizado bate na lista canônica?
        if rn_n in canon_norm:
            real2canon[rn] = canon_norm[rn_n]
            mapped_norm += 1
        else:
            real2canon[rn] = None

    # 6) heurísticas por token
    if heur_token2class:
        for rn, cur in list(real2canon.items()):
            if cur is None:
                s = clean_norm(rn)
                for toks, target in heur_token2class:
                    if any(t in s for t in toks) and target in canonical:
                        real2canon[rn] = target
                        mapped_heur += 1
                        break

    # 7) fuzzy (opcional)
    if auto_fuzzy:
        import difflib
        for rn, cur in list(real2canon.items()):
            if cur is None:
                rn_n = clean_norm(rn)
                close = difflib.get_close_matches(rn_n, canon_names, n=1, cutoff=float(fuzzy_cutoff))
                if close:
                    real2canon[rn] = close[0]
                    mapped_fuzzy += 1

    # 8) LLM (opcional)
    to_classify = [rn for rn in real_names if real2canon[rn] is None and use_llm]
    if to_classify:
        llm_map = classify_ollama_batch(to_classify, canon_names, model=llm_model,
                                        temperature=llm_temp, batch=llm_batch)
        
        for rn in to_classify:
            pred = llm_map.get(rn)
            if not pred:
                continue
            pv = clean_norm(pred)
            if pv in canonical:
                real2canon[rn] = pv
                mapped_llm += 1
                continue
            pv_syn = syn_norm.get(pv)
            if pv_syn in canonical:
                real2canon[rn] = pv_syn
                mapped_llm += 1

    dropped = sorted([rn for rn, v in real2canon.items() if v is None])
    report = {
        "total_raw": len(real_names),
        "mapped_exact": mapped_exact,
        "mapped_norm": mapped_norm,
        "mapped_llm": mapped_llm,
        "mapped_heur": mapped_heur,
        "mapped_fuzzy": mapped_fuzzy,
        "dropped": len(dropped),
        "dropped_samples": dropped[:50],
    }
    return real2canon, report

def _read_yaml(path):
    import yaml, os
    if not path: return {}
    if not os.path.exists(path): 
        raise FileNotFoundError(f"Config não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _apply_cli_overrides(args, cfg):
    cfg.setdefault("paths", {})
    cfg.setdefault("llm", {})
    cfg.setdefault("matching", {})
    cfg.setdefault("normalization", {})
    cfg.setdefault("heuristics", {})
    P = cfg["paths"]; M = cfg["matching"]; L = cfg["llm"]
    if args.xml_root:    P["xml_root"] = args.xml_root
    if args.out_root:    P["out_root"] = args.out_root
    if args.classes:     P["classes"] = args.classes
    if args.img_roots is not None: P["img_roots"] = args.img_roots
    if args.copy_images: P["copy_images"] = True
    if args.resize_long and args.resize_long > 0: P["resize_long"] = int(args.resize_long)
    if args.val_ratio is not None: cfg["paths"]["val_ratio"] = float(args.val_ratio)
    if args.min_box_px is not None: cfg["paths"]["min_box_px"] = int(args.min_box_px)
    if args.seed is not None: cfg["paths"]["seed"] = int(args.seed)
    if args.auto_fuzzy: M["auto_fuzzy"] = True
    if args.fuzzy_cutoff is not None: M["fuzzy_cutoff"] = float(args.fuzzy_cutoff)
    if args.use_llm: 
        L["enabled"] = True
    return cfg

def load_class_map_any(cfg: dict, cli_class_map_path: str, canonical_names: list[str]) -> dict:
    import json, yaml, os
    def _load_mapping_file(path: str) -> dict:
        if not path or not os.path.exists(path): return {}
        if path.lower().endswith((".yml", ".yaml")):
            return yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        return json.load(open(path, "r", encoding="utf-8"))

    cmap_raw = {}

    # 1) CLI tem prioridade
    if cli_class_map_path:
        cmap_raw.update(_load_mapping_file(cli_class_map_path))

    # 2) YAML: aceitar tanto 'normalization.class_map' quanto 'class_map' top-level
    norm_cm = ((cfg.get("normalization") or {}).get("class_map")) or {}
    if isinstance(norm_cm, dict):
        cmap_raw.update(norm_cm)
    if isinstance(cfg.get("class_map"), dict):
        cmap_raw.update(cfg["class_map"])

    # 3) Caminho para arquivo no YAML (paths.class_map | class_map_path)
    path_cfg = (cfg.get("paths", {}) or {}).get("class_map") or cfg.get("class_map_path")
    if path_cfg and not cli_class_map_path:
        cmap_raw.update(_load_mapping_file(path_cfg))

    # 4) Validar e duplicar chaves normalizadas
    valid = set(canonical_names)
    out = {}
    for k, v in (cmap_raw or {}).items():
        if not v or v not in valid:
            continue
        out[str(k)] = v
        out[clean_norm(k)] = v 

    return out

# --------------------- conversor ---------------------
def process_one_xml(xml, args, real2canon, names, name_to_id, labels_dir, images_dir, skipped, unknowns, suggestions, per_class):
    img_path = find_image_for_xml(xml, img_roots=args.img_roots or [])
    if not img_path or not os.path.isfile(img_path):
        skipped["img_not_found"] += 1
        return None
    try:
        with Image.open(img_path) as im:
            W, H = im.size
    except Exception:
        skipped["img_open_fail"] += 1
        return None
    try:
        root = ET.parse(xml).getroot()
    except Exception:
        skipped["xml_parse_fail"] += 1
        return None
    lines = []
    for raw_label, xyxy in parse_boxes_from_xml(root):
        canon = real2canon.get(raw_label)
        if canon is None:
            unknowns[raw_label] += 1
            raw_norm = clean_norm(raw_label)
            close = difflib.get_close_matches(raw_norm, names, n=1, cutoff=float(args.fuzzy_cutoff))
            if close:
                suggestions[raw_label][close[0]] += 1
            continue
        cls_id = name_to_id[canon]
        x1, y1, x2, y2 = xyxy
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if (x2 - x1) < args.min_box_px or (y2 - y1) < args.min_box_px:
            skipped["tiny_box"] += 1
            continue
        xc = (x1 + x2) / 2.0 / W
        yc = (y1 + y2) / 2.0 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        per_class[canon] += 1
    if not lines:
        skipped["no_labels_after_filter"] += 1
        return None
    base = Path(img_path).stem
    write_txt(Path(labels_dir, base + ".txt"), lines)
    if images_dir:
        from shutil import copy2
        dst = Path(images_dir, Path(img_path).name)
        if str(dst.resolve()) != str(Path(img_path).resolve()):
            if int(args.resize_long or 0) > 0:
                try:
                    with Image.open(img_path) as im:
                        W0, H0 = im.size
                        L = max(W0, H0)
                        if L > int(args.resize_long):
                            scale = float(args.resize_long) / float(L)
                            im = im.resize((int(W0 * scale), int(H0 * scale)), Image.BICUBIC)
                        im.save(dst)
                except Exception:
                    copy2(img_path, dst)
            else:
                copy2(img_path, dst)
        return str(dst.resolve())
    else:
        return str(Path(img_path).resolve())
    
def voc_to_yolo_from_config(args, cfg):
    def upd_attr(name, value):
        if value is None:
            return
        if getattr(args, name, None) in (None, [], 0, False):
            setattr(args, name, value)
    upd_attr("xml_root",   cfg_get(cfg, "paths.xml_root"))
    upd_attr("out_root",   cfg_get(cfg, "paths.out_root"))
    upd_attr("classes",    cfg_get(cfg, "paths.classes"))
    upd_attr("img_roots",  cfg_get(cfg, "paths.img_roots"))
    if cfg_get(cfg, "paths.copy_images", None) is not None and not args.copy_images:
        args.copy_images = bool(cfg_get(cfg, "paths.copy_images"))
    if not getattr(args, "resize_long", None):
        args.resize_long = int(cfg_get(cfg, "paths.resize_long", 0) or 0)
    if cfg_get(cfg, "matching.auto_fuzzy", None) is not None:
        args.auto_fuzzy = bool(cfg_get(cfg, "matching.auto_fuzzy"))
    if cfg_get(cfg, "matching.fuzzy_cutoff", None) is not None:
        args.fuzzy_cutoff = float(cfg_get(cfg, "matching.fuzzy_cutoff"))
    if cfg_get(cfg, "llm.enabled", None) is not None:
        args.use_llm = bool(cfg_get(cfg, "llm.enabled"))
    if cfg_get(cfg, "paths.val_ratio", None) is not None:
        args.val_ratio = float(cfg_get(cfg, "paths.val_ratio"))
    if cfg_get(cfg, "paths.min_box_px", None) is not None:
        args.min_box_px = int(cfg_get(cfg, "paths.min_box_px"))
    if cfg_get(cfg, "paths.seed", None) is not None:
        args.seed = int(cfg_get(cfg, "paths.seed"))
        random.seed(args.seed)
    llm_model = cfg_get(cfg, "llm.model", "llama3")
    llm_temp  = float(cfg_get(cfg, "llm.temperature", 0.0) or 0.0)
    llm_batch = int(cfg_get(cfg, "llm.batch", 50) or 50)
    synonyms = cfg_get(cfg, "normalization.synonyms", {}) or {}
    heur_pairs_cfg = cfg_get(cfg, "heuristics.token2class", []) or []
    heur_token2class = []
    for pair in heur_pairs_cfg:
        toks, target = pair
        toks_norm = [clean_norm(t) for t in toks]
        heur_token2class.append((toks_norm, str(target)))
    if not args.xml_root or not args.out_root or not args.classes:
        raise ValueError("Defina xml_root, out_root e classes via --config ou CLI.")
    names = load_names(args.classes)
    name_to_id = {n: i for i, n in enumerate(names)}
    class_map = load_class_map_any(cfg, getattr(args, "class_map", None), names)
    xml_files = sorted(glob.glob(str(Path(args.xml_root, "**/*.xml")), recursive=True))

    if not xml_files:
        raise FileNotFoundError(f"Nenhum XML encontrado em {args.xml_root}")

    real_names_from_xml = set()

    for x in tqdm(xml_files, desc="Coletando rótulos"):
        try:
            for obj in ET.parse(x).getroot().findall("object"):
                nm = obj.findtext("name")
                if nm:
                    real_names_from_xml.add(nm)
        except Exception:
            pass
    real2canon, map_report = build_mapper(
        real_names=sorted(real_names_from_xml),
        canon_names=names,
        synonyms=synonyms,
        class_map=class_map,
        heur_token2class=heur_token2class,
        use_llm=bool(args.use_llm),
        llm_model=llm_model,
        llm_temp=llm_temp,
        llm_batch=llm_batch,
        auto_fuzzy=bool(args.auto_fuzzy),
        fuzzy_cutoff=float(args.fuzzy_cutoff),
    )
    labels_dir = Path(args.out_root, "labels"); ensure_dir(labels_dir)
    images_dir = Path(args.out_root, "images") if args.copy_images else None
    if images_dir: ensure_dir(images_dir)
    splits_dir = Path(args.out_root, "splits"); ensure_dir(splits_dir)
    reports_dir = Path(args.out_root, "reports"); ensure_dir(reports_dir)
    kept_images = []
    per_class = Counter()
    skipped = Counter()
    unknowns = Counter()
    suggestions = defaultdict(Counter)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = []
        for xml in xml_files:
            futures.append(executor.submit(
                process_one_xml, xml, args, real2canon, names, name_to_id,
                labels_dir, images_dir, skipped, unknowns, suggestions, per_class
            ))
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processando VOC→YOLO (paralelo)"):
            result = f.result()
            if result:
                kept_images.append(result)

    random.shuffle(kept_images)
    n_val = max(1, int(len(kept_images) * args.val_ratio)) if kept_images else 0
    train_list = kept_images[n_val:]
    val_list   = kept_images[:n_val]
    write_txt(Path(splits_dir, "train.txt"), train_list)
    write_txt(Path(splits_dir, "val.txt"),   val_list)
    data_yaml = {
        "train": [str(Path(splits_dir, "train.txt").resolve())],
        "val":   [str(Path(splits_dir, "val.txt").resolve())],
        "nc": len(names),
        "names": names
    }
    write_yaml(data_yaml, Path(args.out_root, "data.yaml"))
    if cfg_get(cfg, "outputs.reports", True):
        if unknowns:
            skel = {lbl: "<classe_canonica>" for lbl, _ in unknowns.most_common()}
            with open(Path(reports_dir, "class_map_skeleton.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(skel, f, sort_keys=True, allow_unicode=True)
            with open(Path(reports_dir, "unknown_labels.tsv"), "w", encoding="utf-8") as f:
                f.write("raw_label\tcount\tsuggestion\n")
                for lbl, cnt in unknowns.most_common():
                    sug = ",".join([f"{k}({v})" for k, v in suggestions[lbl].most_common()]) or "-"
                    f.write(f"{lbl}\t{cnt}\t{sug}\n")
        with open(Path(reports_dir, "class_counts.tsv"), "w", encoding="utf-8") as f:
            for k, v in sorted(per_class.items()):
                f.write(f"{k}\t{v}\n")
        with open(Path(reports_dir, "mapper_report.json"), "w", encoding="utf-8") as f:
            json.dump(map_report, f, indent=2, ensure_ascii=False)
    print("\n[OK] VOC→YOLO")
    print("  imagens :", len(kept_images))
    print("  caixas  :", sum(per_class.values()))
    print("  data.yaml:", Path(args.out_root, "data.yaml"))
    print("  mapa    :", {k: map_report[k] for k in ("total_raw","mapped_exact","mapped_norm","mapped_llm","mapped_heur","mapped_fuzzy","dropped")})
    if unknowns:
        print("  ⚠ rótulos desconhecidos:", sum(unknowns.values()), "→", Path(reports_dir, "unknown_labels.tsv"))
    if any(v > 0 for v in (skipped["img_not_found"], skipped["img_open_fail"], skipped["xml_parse_fail"], skipped["tiny_box"], skipped["no_labels_after_filter"])):
        print("  avisos (descartes):", dict(skipped))

def merge_datasets_from_config(args, cfg):
    P = cfg.get("paths", {}) or {}
    M = cfg.get("merge", {}) or {}
    out_root     = args.out_root or P.get("out_root")
    classes_path = args.classes  or P.get("classes")
    copy_images  = args.copy_images if args.copy_images else bool(P.get("copy_images", True))
    resize_long  = args.resize_long if args.resize_long else int(P.get("resize_long", 0))
    respect_splits = bool(M.get("respect_splits", True))
    val_ratio    = float(P.get("val_ratio", cfg.get("val_ratio", 0.1)))
    sources      = M.get("sources") or []
    if not out_root:
        raise ValueError("Defina paths.out_root no YAML ou --out_root na CLI.")
    if not sources or not isinstance(sources, list):
        raise ValueError("Defina merge.sources (lista) no YAML com as pastas dos datasets a unir.")
    if not classes_path:
        first_names = load_names_from_dataset(sources[0])
        if not first_names:
            raise ValueError("Defina paths.classes ou garanta que o primeiro dataset tem data.yaml com 'names'.")
        canonical_names = [str(x) for x in first_names]
    else:
        canonical_names = [str(x) for x in load_names(classes_path)]
    canonical_nc = len(canonical_names)
    out_root = Path(out_root)
    imgs_out = out_root / "images"
    labs_out = out_root / "labels"
    splits_out = out_root / "splits"
    for d in (imgs_out, labs_out, splits_out):
        d.mkdir(parents=True, exist_ok=True)
    if resize_long and resize_long > 0:
        copy_images = True
    merged_train, merged_val = [], []
    global_img_counter = 0
    per_source_stats = []
    for si, ds_root in enumerate(tqdm(sources, desc="Fontes")):
        ds_root = Path(ds_root)
        if not ds_root.exists():
            print(f"[AVISO] Fonte não encontrada: {ds_root}")
            continue
        src_names = load_names_from_dataset(ds_root)
        if not src_names:
            if classes_path:
                src_names = canonical_names[:]
            else:
                raise RuntimeError(f"{ds_root} sem data.yaml(names) e sem classes fornecidas.")
        id_map = build_id_remap(src_names, canonical_names)
        if id_map is None:
            raise RuntimeError(
                f"Incompatibilidade de classes em {ds_root}.\n"
                f"Fonte tem: {src_names}\n"
                f"Canonico : {canonical_names}"
            )
        if respect_splits:
            tr, va = list_images_from_splits(ds_root)
            if not tr and not va:
                imgs = scan_images_from_labels(ds_root)
                random.shuffle(imgs)
                n_val = max(1, int(len(imgs)*val_ratio))
                tr, va = imgs[n_val:], imgs[:n_val]
        else:
            imgs = scan_images_from_labels(ds_root)
            random.shuffle(imgs)
            n_val = max(1, int(len(imgs)*val_ratio))
            tr, va = imgs[n_val:], imgs[:n_val]
        def handle_one(img_path):
            nonlocal global_img_counter
            base = safe_basename(img_path)
            new_base = f"d{si}_{global_img_counter:07d}_{base}"
            global_img_counter += 1
            src_img = Path(img_path)
            ext = src_img.suffix.lower()
            if ext not in IMG_EXTS:
                ext = ".jpg"
            dst_img = imgs_out / f"{new_base}{ext}"
            copy_or_link_image(str(src_img), str(dst_img), copy=copy_images, resize_long=resize_long)
            src_lab = Path(ds_root, "labels", src_img.stem + ".txt")
            dst_lab = labs_out / f"{new_base}.txt"
            if src_lab.exists():
                if all(i==id_map[i] for i in id_map):
                    Path(dst_lab).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        from shutil import copy2
                        copy2(src_lab, dst_lab)
                    except Exception:
                        with open(src_lab, "r", encoding="utf-8") as f:
                            lines = [ln.strip() for ln in f if ln.strip()]
                        with open(dst_lab, "w", encoding="utf-8") as g:
                            g.write("\n".join(lines))
                else:
                    remap_label_file(str(src_lab), str(dst_lab), id_map)
            else:
                with open(dst_lab, "w", encoding="utf-8") as g:
                    g.write("")
            return str(dst_img.resolve())
        tr_out = [handle_one(p) for p in tqdm(tr, desc=f"Treino {ds_root.name}", leave=False)]
        va_out = [handle_one(p) for p in tqdm(va, desc=f"Val {ds_root.name}", leave=False)]
        merged_train.extend(tr_out)
        merged_val.extend(va_out)
        per_source_stats.append({
            "source": str(ds_root),
            "train": len(tr_out),
            "val": len(va_out),
            "nc": len(src_names),
            "id_map_changed": any(id_map[i]!=i for i in id_map)
        })
    random.shuffle(merged_train); random.shuffle(merged_val)
    write_text(splits_out / "train.txt", merged_train)
    write_text(splits_out / "val.txt", merged_val)
    data_yaml = {
        "path": str(out_root.resolve()),
        "train": [str((splits_out/"train.txt").resolve())],
        "val":   [str((splits_out/"val.txt").resolve())],
        "nc": canonical_nc,
        "names": canonical_names
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print("\n[OK] MERGE concluído")
    print("  out_root:", out_root)
    print("  train   :", len(merged_train))
    print("  val     :", len(merged_val))
    print("  classes :", canonical_nc)
    for s in per_source_stats:
        print("   - src:", s["source"], "| train:", s["train"], "| val:", s["val"], "| remap_ids:", s["id_map_changed"])

def main():
    import argparse, random
    ap = argparse.ArgumentParser(
        description="Normalizar dataset VOC para classes canônicas (YOLO) com config YAML, LLM, sinônimos e heurísticas."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("from_voc", help="Converter VOC XML + imagens → YOLO canônico")
    a.add_argument("--config", type=str, default=None, help="YAML de configuração (estilo icons)")
    a.add_argument("--xml_root", type=str, default=None)
    a.add_argument("--out_root", type=str, default=None)
    a.add_argument("--classes", type=str, default=None)
    a.add_argument("--img_roots", nargs="*", default=None, help="pastas onde procurar imagens (recursivo)")
    a.add_argument("--copy_images", action="store_true")
    a.add_argument("--resize_long", type=int, default=0, help="lado maior alvo ao copiar (0=sem resize)")
    a.add_argument("--val_ratio", type=float, default=0.1)
    a.add_argument("--min_box_px", type=int, default=2)
    a.add_argument("--auto_fuzzy", action="store_true")
    a.add_argument("--fuzzy_cutoff", type=float, default=0.92)
    a.add_argument("--use_llm", action="store_true")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--class_map", type=str, default=None,
                help="arquivo JSON/YAML com map 'rótulo_real' -> 'classe_canônica'; sobrescreve o config")
    m = sub.add_parser("merge", help="Unir múltiplos datasets YOLO (real + sintético) em um único out_root")
    m.add_argument("--config", type=str, default=None, help="YAML com paths.merge.sources, paths.out_root, paths.classes")
    m.add_argument("--out_root", type=str, default=None)
    m.add_argument("--classes", type=str, default=None)
    m.add_argument("--copy_images", action="store_true")
    m.add_argument("--resize_long", type=int, default=0, help="lado maior alvo ao copiar (0=sem resize)")
    args = ap.parse_args()
    cfg = _read_yaml(args.config) if args.config else {}
    
    if args.cmd == "from_voc":
        cfg = _apply_cli_overrides(args, cfg)
        seed = args.seed if args.seed is not None else int(cfg.get("paths", {}).get("seed", 0))
        random.seed(seed)
        voc_to_yolo_from_config(args, cfg)
    elif args.cmd == "merge":
        merge_datasets_from_config(args, cfg)

if __name__ == "__main__":
    main()