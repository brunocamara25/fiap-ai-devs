import os, sys, json, csv, argparse, shutil, fnmatch
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import difflib

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

try:
    import ollama  # opcional (se LLM habilitado)
except Exception:
    ollama = None

def norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_").replace("+", "plus")

def path_posix(p: Path) -> str:
    try:
        return p.as_posix()
    except Exception:
        return str(p)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, do_copy: bool):
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy2(src, dst)

def get_provider(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    for prov in ("aws", "azure", "gcp"):
        if prov in parts:
            return prov
    return "unknown"
def load_token_rules(cfg: dict):
    rules = []
    raw = cfg.get("heuristics", {}).get("token2class", {})
    for k, v in raw.items():
        tokens = [norm(x) for x in k.split(",") if x.strip()]
        rules.append((tokens, norm(v)))
    return rules

def infer_category_from_name(stem: str, rules, closed_classes):
    s = norm(stem)
    tokens = s.split("_")
    # 1. Busca por token exato
    for toks, cls in rules:
        if any(tok in tokens for tok in toks):
            return cls
    # 2. Busca por substring
    for toks, cls in rules:
        if any(tok in s for tok in toks):
            return cls
    # 3. Fuzzy match com closed_classes
    best = difflib.get_close_matches(s, closed_classes, n=1, cutoff=0.7)
    if best:
        return best[0]
    return None

def get_level2_folder(p: Path) -> Optional[str]:
    parts = [norm(x) for x in p.parts]
    for i, part in enumerate(parts):
        if part in ("aws", "azure", "gcp") and i + 1 < len(parts):
            return parts[i + 1]
    # Se não houver subpasta, tenta inferir do nome do arquivo
    stem = norm(p.stem)
    if "_" in stem:
        return stem.split("_")[0]
    return stem

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def load_config(path: str) -> dict:
    if not path or not Path(path).exists():
        print("Você deve fornecer --config <arquivo.yml> válido.", file=sys.stderr)
        sys.exit(2)
    if yaml is None:
        print("Instale PyYAML para usar --config: pip install pyyaml", file=sys.stderr)
        sys.exit(2)
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return user

def scan_icons(root: Path) -> List[Path]:
    imgs = []
    for ext in ("*.svg", "*.SVG", "*.png", "*.PNG"):
        imgs.extend(root.rglob(ext))
    seen, out = set(), []
    for p in imgs:
        if p.exists() and p not in seen:
            seen.add(p)
            out.append(p)
    return out

def should_ignore(p: Path, cfg: dict, icons_root: Path) -> bool:
    rel = path_posix(p.relative_to(icons_root))
    for pat in cfg["ignore"].get("folder_globs", []):
        if fnmatch.fnmatch(rel, pat):
            return True
    stem = norm(p.stem)
    for tok in cfg["ignore"].get("name_tokens", []):
        if tok in stem:
            return True
    return False

def load_token_rules(cfg: dict) -> List[Tuple[List[str], str]]:
    rules = []
    raw = cfg.get("heuristics", {}).get("token2class", {})
    for k, v in raw.items():
        tokens = [norm(x) for x in k.split(",") if x.strip()]
        rules.append((tokens, norm(v)))
    return rules

def apply_heuristics(name: str, rules: List[Tuple[List[str], str]]) -> Optional[str]:
    s = norm(name)
    for toks, cls in rules:
        if any(tok in s for tok in toks):
            return cls
    return None

def llm_available() -> bool:
    return ollama is not None

def classify_batch_llm(items: List[Tuple[str, str, str]], allowed: List[str], cfg: dict) -> Dict[str, Tuple[str, float]]:
    out: Dict[str, Tuple[str, float]] = {}
    if not items or not llm_available():
        return out
    model = cfg["llm"].get("model", "llama3")
    temp = float(cfg["llm"].get("temperature", 0.0))
    batch_size = int(cfg["llm"].get("batch_size", 40))
    acceptance = float(cfg["llm"].get("acceptance_threshold", 0.0))
    allowed_sorted = sorted(set(norm(x) for x in allowed))
    force_closed = bool(cfg["unify_folders"].get("force_closed_set", True))
    print(f"[LLM] Classificando {len(items)} itens com modelo '{model}'...")
    for i in tqdm(range(0, len(items), batch_size), desc="Classificando LLM"):
        chunk = items[i:i + batch_size]
        lines = []
        for stem, provider, folder in chunk:
            hint = f" provider={provider}" if cfg["llm"].get("provider_hint", True) else ""
            lines.append(f"- name:{stem} folder:{folder}{hint}")
        prompt = (
            "You are mapping cloud service icon names to ONE canonical class from the CLOSED SET below.\n"
            "Return JSONL lines, one per item, with fields: name, class, confidence (0..1).\n"
            f"ALLOWED_CLASSES: {allowed_sorted}\n"
            "Rules:\n"
            "- Choose exactly one class from ALLOWED_CLASSES.\n"
            "- If uncertain, pick the most likely.\n"
            "- Output ONLY JSON lines (no prose).\n\n"
            "ITEMS:\n" + "\n".join(lines)
        )
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temp}
            )
            text = resp.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[LLM] erro: {e}", file=sys.stderr)
            continue
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                name = str(obj.get("name", "")).strip()
                klass = norm(str(obj.get("class", "")).strip())
                conf = float(obj.get("confidence", 0.0))
                if not name:
                    continue
                if force_closed and klass not in allowed_sorted:
                    continue
                if conf < acceptance:
                    continue
                out[name] = (klass, conf)
            except Exception:
                if "," in raw:
                    name, klass = [x.strip() for x in raw.split(",", 1)]
                    klass = norm(klass)
                    if (not force_closed) or (klass in allowed_sorted):
                        out[name] = (klass, 0.5)
                else:
                    continue
    return out

def map_folders_llm(folders: List[Tuple[str, str]], allowed: List[str], cfg: dict) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    if not folders or not llm_available():
        return out
    model = cfg["llm"].get("model", "llama3")
    temp = float(cfg["llm"].get("temperature", 0.0))
    batch_size = int(cfg["llm"].get("batch_size", 40))
    allowed_sorted = sorted(set(norm(x) for x in allowed))
    force_closed = bool(cfg["unify_folders"].get("force_closed_set", True))
    for i in range(0, len(folders), batch_size):
        chunk = folders[i:i + batch_size]
        lines = [f"- provider:{prov} folder:{fld}" for prov, fld in chunk]
        prompt = (
            "You are normalizing cloud provider SECOND-LEVEL folders into ONE canonical class from the CLOSED SET below.\n"
            "Return JSONL lines, one per item, with fields: provider, folder, class.\n"
            f"CLOSED_SET: {allowed_sorted}\n"
            "Rules: choose exactly one class from CLOSED_SET. ONLY JSON lines.\n\n"
            "ITEMS:\n" + "\n".join(lines)
        )
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temp}
            )
            text = resp.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"[LLM] erro folder-unify: {e}", file=sys.stderr)
            continue
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                prov = norm(str(obj.get("provider", "")).strip())
                fld = norm(str(obj.get("folder", "")).strip())
                klass = norm(str(obj.get("class", "")).strip())
                if not prov or not fld:
                    continue
                if force_closed and klass not in allowed_sorted:
                    continue
                out[(prov, fld)] = klass
            except Exception:
                parts = [x.strip() for x in raw.split(",")]
                if len(parts) == 3:
                    prov, fld, klass = parts
                    prov, fld, klass = norm(prov), norm(fld), norm(klass)
                    if (not force_closed) or (klass in allowed_sorted):
                        out[(prov, fld)] = klass
    return out

def write_classes_txt(out_dir: Path, classes_used: List[str]):
    (out_dir / "classes.txt").write_text("\n".join(classes_used) + "\n", encoding="utf-8")

def write_classes_yaml(out_dir: Path, classes_used: List[str]):
    try:
        import yaml
        data = {"classes": classes_used}
        with open(out_dir / "classes.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)
    except Exception:
        pass

def write_icon_map_csv(out_dir: Path, rows: List[dict]):
    with open(out_dir / "icon_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "provider", "folder", "stem", "class"])
        w.writeheader()
        w.writerows(rows)

def write_icon_map_json(out_dir: Path, rows: List[dict]):
    with open(out_dir / "icon_map.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def write_unknown_csv(out_dir: Path, rows: List[dict]):
    with open(out_dir / "unknown_icons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "provider", "folder", "stem"])
        w.writeheader()
        w.writerows(rows)

def write_folder_unify_json(out_dir: Path, unify_map: Dict[str, Dict[str, str]]):
    with open(out_dir / "folder_unify.json", "w", encoding="utf-8") as f:
        json.dump(unify_map, f, ensure_ascii=False, indent=2)

def write_class_stats_json(out_dir: Path, stats: Dict[str, int]):
    with open(out_dir / "class_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def validate_icon_coverage(icons: List[Path], unify_map: Dict[str, Dict[str, str]], closed: List[str], rules, out_dir: Path):
    """
    Valida se todos os ícones estão cobertos pelo mapeamento do folder_unify.json.
    Gera um arquivo 'uncovered_icons.csv' com os ícones não classificados.
    """
    uncovered = []
    for p in icons:
        prov = get_provider(p)
        fld = get_level2_folder(p) or ""
        stem = norm(p.stem)
        assigned = None

        # Tenta mapear usando o unify_map (sem ifs de provider)
        prov_map = unify_map.get(prov, {})
        # Para GCP, pode ser por nome do arquivo, para outros, por pasta
        assigned = prov_map.get(fld) or prov_map.get(p.name) or prov_map.get(stem)

        # Se não achou, tenta heurística
        if not assigned:
            assigned = infer_category_from_name(stem, rules, closed)

        if assigned not in closed:
            uncovered.append({
                "path": str(p),
                "provider": prov,
                "folder": fld,
                "stem": p.stem
            })

    if uncovered:
        print(f"\n[VALIDAÇÃO] {len(uncovered)} ícones NÃO cobertos pelo mapeamento. Veja 'uncovered_icons.csv'.")
        write_unknown_csv(out_dir, uncovered)
    else:
        print("\n[VALIDAÇÃO] Todos os ícones estão cobertos pelo mapeamento! 🎉")
        
def main():
    ap = argparse.ArgumentParser(description="Classifica ícones de cloud em classes funcionais (com --config YAML).")
    ap.add_argument("--config", required=True, help="YAML de configuração (obrigatório)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    icons_root = Path(cfg["paths"]["icons_root"]).resolve()
    out_dir = Path(cfg["paths"]["out"]).resolve()
    ensure_dir(out_dir)

    # Limpa a pasta de unknow (se existir)
    unknown_dir = out_dir / "icons" / "unknown"
    if unknown_dir.exists() and unknown_dir.is_dir():
        print(f"[INFO] Limpando pasta {unknown_dir}")
        shutil.rmtree(unknown_dir)

    # Limpa o arquivo unknown_icons.csv (opcional)
    unknown_csv = out_dir / "unknown_icons.csv"
    if unknown_csv.exists():
        unknown_csv.unlink()
        
    closed = [norm(x) for x in cfg["classes"]["closed_set"]]
    if cfg["classes"].get("include_extras", False):
        closed += [norm(x) for x in cfg["classes"].get("extras", [])]
    closed = sorted(set(closed))

    all_icons = scan_icons(icons_root)
    icons: List[Path] = []
    for p in tqdm(all_icons, desc="Filtrando ícones"):
        try:
            if should_ignore(p, cfg, icons_root):
                continue
            icons.append(p)
        except Exception:
            continue

    provider_folders = defaultdict(set)
    for p in icons:
        prov = get_provider(p)
        fld = get_level2_folder(p)
        if prov != "unknown" and fld:
            provider_folders[prov].add(fld)

    unify_map: Dict[str, Dict[str, str]] = defaultdict(dict)
    cache_file = (out_dir / cfg["unify_folders"].get("cache_file", "folder_unify.json")).resolve()
    if cache_file.exists():
        try:
            unify_map = defaultdict(dict, json.loads(read_text(cache_file)))
        except Exception:
            unify_map = defaultdict(dict)

    rules = load_token_rules(cfg)
    remaining_to_map = []
    if cfg["unify_folders"].get("enabled", True):
        for prov, folders in tqdm(provider_folders.items(), desc="Unificando folders"):
            for fld in sorted(folders):
                if fld in unify_map.get(prov, {}):
                    continue
                guess = apply_heuristics(fld, rules)
                if guess in closed:
                    unify_map[prov][fld] = guess
                else:
                    remaining_to_map.append((prov, fld))
        if remaining_to_map and cfg["llm"].get("enabled", False):
            if not llm_available():
                print("[LLM] Ollama não disponível; pulando unificação de pastas.")
            else:
                folder_llm = map_folders_llm(remaining_to_map, closed, cfg)
                for (prov, fld), klass in folder_llm.items():
                    if klass in closed:
                        unify_map[prov][fld] = klass
        write_folder_unify_json(out_dir, unify_map)

    class_to_files: Dict[str, List[Path]] = defaultdict(list)
    icon_rows, unknown_rows = [], []
    unknown_for_llm: List[Tuple[str, str, str, Path]] = []
    for p in icons:
        prov = get_provider(p)
        fld = get_level2_folder(p) or ""
        stem = norm(p.stem)
        assigned: Optional[str] = None
        if fld and prov in unify_map and fld in unify_map[prov]:
            assigned = unify_map[prov][fld]
        if not assigned:
            assigned = infer_category_from_name(stem, rules, closed)
        if assigned in closed:
            class_to_files[assigned].append(p)
            icon_rows.append({"path": str(p), "provider": prov, "folder": fld, "stem": p.stem, "class": assigned})
        else:
            unknown_rows.append({"path": str(p), "provider": prov, "folder": fld, "stem": p.stem})
            unknown_for_llm.append((p.stem, prov, fld, p))

    # Após classificar com LLM:
    if cfg["llm"].get("enabled", False) and unknown_for_llm:
        if not llm_available():
            print("[LLM] Ollama não disponível; pulando classificação de desconhecidos.")
        else:
            batch_payload = [(stem, prov, fld) for (stem, prov, fld, _) in unknown_for_llm]
            preds = classify_batch_llm(batch_payload, closed, cfg)
            remaining = []
            new_unknown_rows = []
            for (stem, prov, fld, p), unk_row in zip(unknown_for_llm, unknown_rows):
                pair = preds.get(norm(stem))
                if pair:
                    klass, _ = pair
                    if klass in closed:
                        class_to_files[klass].append(p)
                        icon_rows.append({"path": str(p), "provider": prov, "folder": fld, "stem": p.stem, "class": klass})
                        unify_map[prov][fld] = klass
                        continue
                remaining.append((stem, prov, fld, p))
                new_unknown_rows.append(unk_row)
            unknown_for_llm = remaining
            unknown_rows = new_unknown_rows
            # Salve o novo unify_map
            write_folder_unify_json(out_dir, unify_map)

    copy_files = bool(cfg["paths"].get("copy_files", False))
    by_class_dir = out_dir / "icons"
    ensure_dir(by_class_dir)
    classes_used = sorted(class_to_files.keys())
    for cls, files in tqdm(class_to_files.items(), desc="Copiando por classe"):
        cls_dir = by_class_dir / cls
        ensure_dir(cls_dir)
        for src in files:
            dst = cls_dir / src.name
            link_or_copy(src, dst, copy_files)

    emits = set(cfg["outputs"].get("emit", []))
    if "classes_txt" in emits:
        write_classes_txt(out_dir, classes_used)
    if "classes_yaml" in emits:
        write_classes_yaml(out_dir, classes_used)
    if "icon_map_csv" in emits:
        write_icon_map_csv(out_dir, icon_rows)
    if "icon_map_json" in emits:
        write_icon_map_json(out_dir, icon_rows)
    if "unknown_icons_csv" in emits and unknown_rows:
        write_unknown_csv(out_dir, unknown_rows)
    if "folder_unify_json" in emits:
        write_folder_unify_json(out_dir, unify_map)
    if "class_stats_json" in emits:
        stats = Counter([r["class"] for r in icon_rows])
        write_class_stats_json(out_dir, dict(sorted(stats.items(), key=lambda x: (-x[1], x[0]))))

    validate_icon_coverage(icons, unify_map, closed, rules, out_dir)
    
    total = len(icons)
    classified = sum(len(v) for v in class_to_files.values())
    unclassified = len(unknown_rows)
    
    print(f"\n[RESUMO]")
    print(f" total ícones : {total}")
    print(f" classificados : {classified}")
    print(f" não classificados : {unclassified}")
    if classes_used:
        print(f" classes usadas: {', '.join(classes_used)}")
    print("\nArquivos gerados em:", out_dir)
    print(" - classes.txt / classes.yaml")
    print(" - icon_map.csv / icon_map.json")
    print(" - icons/<classe>/")
    if unclassified > 0:
        print(" - unknown_icons.csv (existem ícones não classificados!)")
    else:
        print("Sem desconhecidos 🎉")
    print(" - folder_unify.json")
    print(" - class_stats.json")

if __name__ == "__main__":
    main()