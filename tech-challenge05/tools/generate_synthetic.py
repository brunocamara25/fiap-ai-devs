import os, re, io, math, glob, json, random, argparse
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import yaml
from io import BytesIO

try:
    import cairosvg
    HAS_CAIRO = True
except Exception:
    HAS_CAIRO = False

# ==========================
# Utils básicos
# ==========================
def clean_name(name: str) -> str:
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r'\b(icon|icons|logo|logos|icon_azure|azure_icon|aws_icon|gcp_icon)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[_\-\s]+', '_', name).strip('_').lower()
    return name

def rand_pos(W, H, iw, ih, margin=32):
    return (
        random.randint(margin, max(margin, W - iw - margin)),
        random.randint(margin, max(margin, H - ih - margin))
    )

def draw_dashed(draw, xy, width=2, fill=(0,0,0), dash=(8,8)):
    x1,y1,x2,y2 = xy
    total = math.hypot(x2-x1, y2-y1)
    if total <= 0: return
    dx,dy = (x2-x1)/total, (y2-y1)/total
    d_on, d_off = dash
    dist = 0.0
    while dist < total:
        s, e = dist, min(dist + d_on, total)
        sx,sy = x1 + dx*s, y1 + dy*s
        ex,ey = x1 + dx*e, y1 + dy*e
        draw.line((sx,sy,ex,ey), fill=fill, width=width)
        dist += d_on + d_off

def draw_arrow(draw: ImageDraw.ImageDraw, p1, p2, width=3, fill=(0,0,0)):
    draw.line((*p1,*p2), width=width, fill=fill)
    ang = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
    L = 10 + random.randint(0,6)
    a1 = (p2[0]-L*math.cos(ang-0.5), p2[1]-L*math.sin(ang-0.5))
    a2 = (p2[0]-L*math.cos(ang+0.5), p2[1]-L*math.sin(ang+0.5))
    draw.polygon([p2, a1, a2], fill=fill)
    xs = [p1[0], p2[0], a1[0], a2[0]]
    ys = [p1[1], p2[1], a1[1], a2[1]]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

def get_font(size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    random.shuffle(font_paths)
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def random_label_style(label):
    style = random.choice(['upper','lower','capitalize','title','break','plain'])
    if style == 'upper': return label.upper()
    if style == 'lower': return label.lower()
    if style == 'capitalize': return label.capitalize()
    if style == 'title': return label.title()
    if style == 'break':
        parts = label.replace('_',' ').split()
        if len(parts)>1:
            i = random.randint(1, len(parts)-1)
            return ' '.join(parts[:i])+'\n'+' '.join(parts[i:])
    return label.replace('_',' ')

def paste_with_shadow(canvas: Image.Image, icon: Image.Image, xy, alpha=120):
    x,y = xy
    shadow = Image.new("RGBA", icon.size, (0,0,0,0))
    sh = Image.new("RGBA", icon.size, (0,0,0,alpha))
    shadow.paste(sh, (2,2))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
    canvas.alpha_composite(shadow, (x,y))
    canvas.alpha_composite(icon, (x,y))

def jitter_icon(pil_rgba: Image.Image):
    arr = np.array(pil_rgba).astype(np.float32)
    mul = np.clip(np.random.normal(1.0, 0.05, size=(1,1,4)), 0.85, 1.15)
    mul[...,3] = 1.0
    arr = np.clip(arr * mul, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)  # Pillow infer mode
    if random.random() < 0.5:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img

def jpeg_compress(pil_img: Image.Image, q_min=40, q_max=95):
    q = random.randint(q_min, q_max)
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    out = Image.open(buf).convert("RGBA")
    return out

def add_noise(pil_img: Image.Image, std=6.0):
    arr = np.array(pil_img).astype(np.int16)
    noise = np.random.normal(0, std, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGBA")

def motion_blur_pil(pil_img: Image.Image, k=5):
    """
    Motion blur robusto para Pillow:
    - garante k ímpar e dentro de [3..31]
    - converte kernel para lista de floats nativos
    - fallback para BoxBlur se o Kernel reclamar
    """
    # força ímpar em faixa segura
    k = int(k)
    if k < 3: k = 3
    if k > 31: k = 31
    if k % 2 == 0: k += 1

    # horizontal (50%) ou vertical (50%)
    if random.random() < 0.5:
        kernel = [[0.0]*k for _ in range(k)]
        for j in range(k):
            kernel[k//2][j] = 1.0
    else:
        kernel = [[0.0]*k for _ in range(k)]
        for i in range(k):
            kernel[i][k//2] = 1.0

    # normaliza
    s = float(k)
    flat = [v/s for row in kernel for v in row]

    rgb = pil_img.convert("RGB")
    try:
        # Alguns builds do Pillow dão "bad kernel size" aleatoriamente;
        # esse try/except cobre e cai num BoxBlur leve.
        rgb = rgb.filter(ImageFilter.Kernel((k, k), flat, scale=1.0))
    except Exception:
        # Fallback visualmente parecido (não direcional)
        # dá um efeito mínimo só pra não ficar 100% “limpo”
        rgb = rgb.filter(ImageFilter.BoxBlur(radius=max(1, (k//2)-1)))

    return rgb.convert("RGBA")

def perspective_warp(pil_img: Image.Image, max_warp=0.06):
    w,h = pil_img.size
    dx = int(w*max_warp*random.uniform(-1,1))
    dy = int(h*max_warp*random.uniform(-1,1))
    src = [(0,0), (w,0), (w,h), (0,h)]
    dst = [(0+dx,0), (w+dx,0+dy), (w, h), (0, h+dy)]
    return pil_img.transform((w,h), Image.QUAD, sum(dst,()), Image.BICUBIC)

def occlude_rect(draw, box, color=(255,255,255,200)):
    x1,y1,x2,y2 = box
    w = x2-x1; h = y2-y1
    ow = int(w * random.uniform(0.2, 0.5))
    oh = int(h * random.uniform(0.2, 0.5))
    ox = random.randint(x1, max(x1, x2 - ow))
    oy = random.randint(y1, max(y1, y2 - oh))
    draw.rectangle([ox,oy,ox+ow,oy+oh], fill=color)

def open_icon_any(path: Path, svg_px: int = 160) -> Image.Image:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".svg":
        if not HAS_CAIRO:
            raise RuntimeError("Precisa de cairosvg para rasterizar SVG (pip install cairosvg)")
        png_bytes = cairosvg.svg2png(url=str(p), output_width=svg_px, output_height=svg_px)
        return Image.open(BytesIO(png_bytes)).convert("RGBA")
    # png/jpg/etc
    return Image.open(p).convert("RGBA")

def get_provider_from_symlink(path_str: str) -> str:
    try:
        real = Path(path_str).resolve()
    except Exception:
        real = Path(path_str)
    parts = [p.lower() for p in real.parts]
    for prov in ("aws","azure","gcp"):
        if prov in parts: return prov
    return "unknown"

def inside(rect, box):
    x1,y1,x2,y2 = rect
    a1,b1,a2,b2 = box
    return (a1>=x1 and b1>=y1 and a2<=x2 and b2<=y2)

# ==========================
# Anti-overlap helpers
# ==========================
def _inflate(x1,y1,x2,y2, d=0):
    return (x1-d, y1-d, x2+d, y2+d)

def _overlap(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def _free(rect, placed):
    return all(not _overlap(rect, p) for p in placed)

def _inside_canvas(rect, W, H):
    x1,y1,x2,y2 = rect
    return x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H

def _measure_multiline(draw, text, font):
    bb = draw.multiline_textbbox((0,0), text, font=font, spacing=2)
    return (bb[2]-bb[0], bb[3]-bb[1])

def place_text(draw, text, xy, font):
    tx, ty = xy
    draw.multiline_text((tx, ty), text, fill=(0,0,0), font=font, spacing=2)

def choose_label_bbox(draw, label, icon_rect, font, label_gap, placed, W, H, min_gap):
    x1,y1,x2,y2 = icon_rect
    tw, th = _measure_multiline(draw, label, font)

    cand = []
    cand.append((x1 + (x2-x1 - tw)//2, y1 - label_gap - th, x1 + (x2-x1 - tw)//2 + tw, y1 - label_gap))
    cand.append((x1 + (x2-x1 - tw)//2, y2 + label_gap,         x1 + (x2-x1 - tw)//2 + tw, y2 + label_gap + th))
    cand.append((x1 - label_gap - tw,   y1 + (y2-y1 - th)//2,  x1 - label_gap,             y1 + (y2-y1 - th)//2 + th))
    cand.append((x2 + label_gap,        y1 + (y2-y1 - th)//2,  x2 + label_gap + tw,        y1 + (y2-y1 - th)//2 + th))
    random.shuffle(cand)

    for cx1,cy1,cx2,cy2 in cand:
        rect = (cx1,cy1,cx2,cy2)
        if _inside_canvas(rect, W, H) and _free(_inflate(*rect, min_gap//2), placed):
            return rect

    # fallback: dentro do ícone (parte de baixo)
    ibw, ibh = (x2-x1), (y2-y1)
    if th + 6 < ibh:
        rx1 = x1 + (ibw - tw)//2
        ry1 = y2 - th - 3
        rect = (rx1, ry1, rx1 + tw, ry1 + th)
        if _free(_inflate(*rect, min_gap//4), placed):
            return rect
    return None

# ==========================
# Classes e ícones
# ==========================
def load_classes(path: Optional[str]) -> List[str]:
    if not path:
        raise RuntimeError("Forneça --classes_file (um por linha) ou um YAML com 'names'.")
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de classes não encontrado: {path}")

    # aceita .txt (uma por linha) ou .yaml (names: [...])
    if path.lower().endswith((".yaml", ".yml")):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if not names: raise ValueError(f"YAML sem 'names': {path}")
        return [str(n).strip() for n in names if str(n).strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if not names: raise ValueError(f"Arquivo vazio: {path}")
        return names

def load_icons(dir: str, svg_px: int = 160):
    """
    Lê icons/<classe>/*.{svg,png,jpg}
    Retorna (classes:list[str], class_to_icons:dict[classe->list[icon dict]])
    """
    base = Path(dir)
    class_to_icons = {}
    classes = []
    for cls_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        files = []
        for ext in ("*.svg","*.png","*.jpg","*.jpeg"):
            files.extend(cls_dir.glob(ext))
        bucket = []
        for p in files:
            try:
                img = open_icon_any(p, svg_px=svg_px)
                raw = p.stem
                bucket.append({ "path": str(p), "name": clean_name(raw), "display": raw, "img": img })
            except Exception:
                continue
        if bucket:
            classes.append(cls)
            class_to_icons[cls] = bucket
    return classes, class_to_icons


def canonicalize_class_name(value, classes_set, normalized_map):
    if not value: return None
    candidate = value.strip()
    if candidate in classes_set: return candidate
    return normalized_map.get(clean_name(candidate))

def classify_ollama(icon_name, classes) -> str:
    try:
        import ollama
    except Exception:
        return ""
    classes_prompt = "\n".join(classes)
    prompt = (
        "Classifique o ícone de serviço de nuvem informado na lista de classes fornecida.\n"
        "Responda usando exatamente um dos nomes abaixo (sensível a sublinhados).\n"
        "Classes disponíveis:\n"
        f"{classes_prompt}\n\n"
        f"Ícone: '{icon_name}'.\n"
        "Forneça somente o nome da classe escolhida."
    )
    try:
        resp = ollama.chat(
            model='llama3',
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        return resp['message']['content'].strip()
    except Exception:
        return ""

def load_class_map(class_map_json, use_llm, icons, classes, default_class):
    classes_set = set(classes)
    normalized_map = {clean_name(cls): cls for cls in classes}
    icon_to_class: Dict[str,str] = {}

    # 1) arquivo de mapeamento manual (opcional)
    if class_map_json and os.path.exists(class_map_json):
        with open(class_map_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            canonical = canonicalize_class_name(v, classes_set, normalized_map)
            if canonical:
                icon_to_class[clean_name(k)] = canonical

    # 2) tentativas por nome limpo
    for ic in icons:
        key = ic["name"]
        if key in icon_to_class: continue
        canonical = canonicalize_class_name(ic["name"], classes_set, normalized_map)
        if not canonical:
            canonical = canonicalize_class_name(ic.get("display",""), classes_set, normalized_map)
        if canonical:
            icon_to_class[key] = canonical

    # 3) LLM (opcional)
    if use_llm:
        print(f"[INFO] Classificando {len(icons)} ícones com Llama3 (Ollama)...")
        for ic in tqdm(icons, desc="Classificando ícones c/ Llama3"):
            key = ic["name"]
            if key in icon_to_class: continue
            predicted = classify_ollama(ic["display"], classes)
            canonical = canonicalize_class_name(predicted, classes_set, normalized_map)
            icon_to_class[key] = canonical if canonical else default_class

    # 4) fallback default
    for ic in icons:
        icon_to_class.setdefault(ic["name"], default_class)
    return icon_to_class

def build_icon_usage_list(icons, icon_to_class, min_uses=8):
    # garante que todos os ícones mapeados apareçam algumas vezes
    icon_list = []
    for ic in icons:
        key = ic["name"]
        if key in icon_to_class:
            icon_list.extend([ic] * min_uses)
    random.shuffle(icon_list)
    return icon_list

# ==========================
# Geração
# ==========================
def main():
    ap = argparse.ArgumentParser(description="Gerador de diagramas sintéticos para treinamento YOLO (diagramas de arquitetura).")
    ap.add_argument("--icons", type=str, required=True, help="Pasta com PNGs de ícones.")
    ap.add_argument("--classes_file", type=str, required=True, help="TXT (uma por linha) ou YAML (names: [...]).")
    ap.add_argument("--out", type=str, default="synthetic_out", help="Pasta de saída.")
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--canvas", type=str, default="1600x1000")
    ap.add_argument("--seed", type=int, default=42)
    # cena
    ap.add_argument("--min_nodes", type=int, default=8)
    ap.add_argument("--max_nodes", type=int, default=20)
    ap.add_argument("--p_negative", type=float, default=0.1)
    # colisão / densidade
    ap.add_argument("--min_gap", type=int, default=24, help="espaço mínimo entre elementos em px")
    ap.add_argument("--label_gap", type=int, default=8, help="distância texto↔ícone")
    ap.add_argument("--retries", type=int, default=200, help="tentativas por ícone")
    ap.add_argument("--max_fill", type=float, default=0.30, help="limite fração de área ocupada (0-1)")
    # domínio real / augs
    ap.add_argument("--skew_max_deg", type=float, default=3.0)
    ap.add_argument("--persp_max", type=float, default=0.02)
    ap.add_argument("--p_perspective", type=float, default=0.2)
    ap.add_argument("--p_downup", type=float, default=0.2)
    ap.add_argument("--p_motion_blur", type=float, default=0.15)
    ap.add_argument("--blur_sigma", type=float, default=0.3)
    ap.add_argument("--noise_std", type=float, default=2.0)
    ap.add_argument("--jpeg_min", type=int, default=50)
    ap.add_argument("--jpeg_max", type=int, default=95)
    ap.add_argument("--p_jpeg", type=float, default=0.85)
    ap.add_argument("--p_noise", type=float, default=0.6)
    ap.add_argument("--p_occlude", type=float, default=0.15)
    # mapeamento classes
    ap.add_argument("--class_map_json", type=str, default="", help="JSON opcional: nome_icone->classe")
    ap.add_argument("--default_class", type=str, default="missing")
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--min_uses", type=int, default=8)
    # splits/data.yaml
    ap.add_argument("--val_ratio", type=float, default=0.1, help="fração p/ validação")
    ap.add_argument("--svg_px", type=int, default=160, help="Tamanho base da rasterização de SVG.")
    # flags novas
    ap.add_argument("--emit_graph", action="store_true", help="Salva grafo JSON por imagem.")
    ap.add_argument("--arrow_class", type=str, default="arrow", help="Nome da classe p/ setas (se existir em classes).")
    ap.add_argument("--boundary_class", type=str, default="boundary", help="Nome da classe p/ caixas de boundary (se existir em classes).")

    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    
    # pastas
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    if args.emit_graph:
        (out / "graphs").mkdir(parents=True, exist_ok=True)

    # canvas
    W, H = map(int, args.canvas.lower().split("x"))

    # classes + ids
    classes = load_classes(args.classes_file)
    name_to_id = {n: i for i, n in enumerate(classes)}

    arrow_cid = name_to_id.get(args.arrow_class, None)
    boundary_cid = name_to_id.get(args.boundary_class, None)

    # Defina default_class corretamente:
    if args.default_class in name_to_id:
        default_class = args.default_class
    else:
        default_class = classes[0]  # fallback para a primeira classe

    # Lê diretamente a árvore por classe (output-icons/by_class)
    classes_found, class_to_icons = load_icons(args.icons, svg_px=args.svg_px)

    # Mantém a ordem das classes do arquivo (e filtra as inexistentes)
    class_to_icons = {c: class_to_icons[c] for c in classes if c in class_to_icons}

    # Checagens
    if not class_to_icons:
        raise RuntimeError(f"Nenhum ícone encontrado em formato by_class dentro de: {args.icons}")
    available_classes = [c for c, items in class_to_icons.items() if items]
    if not available_classes:
        raise RuntimeError("Nenhuma classe possui ícones em by_class.")
    
    # Para reaproveitar o pipeline atual:
    icons = [ic for c in available_classes for ic in class_to_icons[c]]
    icon_to_class = {}
    for c in available_classes:
        for ic in class_to_icons[c]:
            icon_to_class[ic["name"]] = c
    if not icons:
        raise RuntimeError(f"Nenhum PNG encontrado em: {args.icons}")
    
    # garantir uso de todos
    icon_usage_list = build_icon_usage_list(icons, icon_to_class, min_uses=args.min_uses)
    total_imgs = max(args.num, len(icon_usage_list))

    img_paths = []

    for i in tqdm(range(total_imgs), desc="Gerando diagramas sintéticos"):
        nodes = []          # [{"id", "class", "bbox", "label", "provider"}]
        edges = []          # [{"source","target","kind"}]
        ocr_items = []      # [{"text","bbox"}]
        panel_rects = []    # listas de boundaries desenhados
        # fundo
        img = Image.new("RGBA", (W,H), (255,255,255,255))
        draw = ImageDraw.Draw(img)

        # painéis/cards
        if random.random() < 0.9:
            for _ in range(random.randint(1,3)):
                px = random.randint(40, W-400)
                py = random.randint(40, H-300)
                pw = random.randint(300, 700)
                ph = random.randint(220, 480)
                rect_panel = (px,py,px+pw,py+ph)
                panel_rects.append(rect_panel)
                if random.random() < 0.5:
                    draw.rectangle([px,py,px+pw,py+ph], outline=(140,140,140,255), width=2)
                else:
                    draw_dashed(draw, (px,py,px+pw,py), width=2, fill=(160,160,160))
                    draw_dashed(draw, (px,py+ph,px+pw,py+ph), width=2, fill=(160,160,160))
                    draw_dashed(draw, (px,py,px,py+ph), width=2, fill=(160,160,160))
                    draw_dashed(draw, (px+pw,py,px+pw,py+ph), width=2, fill=(160,160,160))
                if random.random() < 0.35:
                    ImageDraw.Draw(img).rectangle([px+1,py+1,px+pw-1,py+ph-1], fill=(0,0,0,50))
                if boundary_cid is not None:
                    x1,y1,x2,y2 = rect_panel
                    xc = ((x1+x2)/2.0)/W; yc = ((y1+y2)/2.0)/H
                    bw = (x2-x1)/W;       bh = (y2-y1)/H
                    yolo_lines.append(f"{boundary_cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        is_negative = random.random() < args.p_negative
        placed: List[Tuple[int,int,int,int]] = []  # retângulos "ocupados" (com gap)
        boxes: List[Tuple[int,int,int,int,str]] = []
        centers: List[Tuple[int,int]] = []
        area_used = 0.0
        canvas_area = W * H

        n_nodes = 0 if is_negative else random.randint(args.min_nodes, args.max_nodes)
        for _ in range(n_nodes):
            if area_used / canvas_area > args.max_fill:
                break

            # escolhe ícone
            if icon_usage_list:
                ii = icon_usage_list.pop()
                cls_name = icon_to_class[ii["name"]]
            else:
                cls_name = random.choice(available_classes)
                ii = random.choice(class_to_icons[cls_name])

            icon = jitter_icon(ii["img"])
            scale = random.uniform(0.28, 0.55)
            iw, ih = icon.size
            iw2, ih2 = int(iw*scale), int(ih*scale)
            if iw2 < 12 or ih2 < 12:
                continue
            icon_r = icon.resize((iw2, ih2), resample=Image.BICUBIC)

            # tenta posição sem colisão (com gap)
            placed_ok = False
            icon_rect = None
            for _try in range(args.retries):
                x, y = rand_pos(W, H, iw2, ih2)
                rect = (x, y, x+iw2, y+ih2)
                if _free(_inflate(*rect, args.min_gap), placed):
                    icon_rect = rect
                    placed_ok = True
                    break
            if not placed_ok:
                continue

            # label
            label = random_label_style(ii["display"])
            font = get_font(random.randint(18, 28))
            label_rect = choose_label_bbox(
                draw=draw, label=label, icon_rect=icon_rect, font=font,
                label_gap=args.label_gap, placed=placed, W=W, H=H, min_gap=args.min_gap
            )

            # desenha ícone + oclusão leve (opcional)
            paste_with_shadow(img, icon_r, (icon_rect[0], icon_rect[1]))
            if random.random() < args.p_occlude:
                occlude_rect(draw, icon_rect, color=(255,255,255, random.randint(120,180)))

            # desenha label se couber
            if label_rect is not None:
                place_text(draw, label, (label_rect[0], label_rect[1]), font=font)
                placed.append(_inflate(*label_rect, args.min_gap//2))

            placed.append(_inflate(*icon_rect, args.min_gap))
            boxes.append((*icon_rect, cls_name))
            node_id = len(nodes)
            nodes.append({
                "id": node_id,
                "class": cls_name,
                "bbox": [icon_rect[0], icon_rect[1], icon_rect[2], icon_rect[3]],
                "label": label if label_rect is not None else "",
                "provider": get_provider_from_symlink(ii["path"])
            })

            if label_rect is not None:
                ocr_items.append({
                    "text": label,
                    "bbox": [label_rect[0], label_rect[1], label_rect[2], label_rect[3]]
                })

            centers.append((( (icon_rect[0]+icon_rect[2])//2, (icon_rect[1]+icon_rect[3])//2 ), node_id))            
            area_used += (icon_rect[2]-icon_rect[0]) * (icon_rect[3]-icon_rect[1])

        # conectores
        n_conn = random.randint(max(1, len(centers)//2), len(centers)+2)
        for _ in range(n_conn):
            if len(centers) < 2: break
            (a_pt, a_id), (b_pt, b_id) = random.sample(centers, 2)
            width = random.randint(2,5)
            if random.random() < 0.5:
                # sólido
                bbox_arrow = draw_arrow(draw, a_pt, b_pt, width=width, fill=(0,0,0))
            else:
                # tracejado + cabeça (fazemos dashed na linha e cabeça sólida no fim)
                draw_dashed(draw, (*a_pt,*b_pt), width=width, fill=(0,0,0), dash=(10,8))
                bbox_arrow = draw_arrow(draw, a_pt, b_pt, width=max(2,width-1), fill=(0,0,0))

            edges.append({"source": a_id, "target": b_id, "kind": "arrow"})

            if arrow_cid is not None:
                x1,y1,x2,y2 = bbox_arrow
                xc = ((x1+x2)/2.0)/W; yc = ((y1+y2)/2.0)/H
                bw = (x2-x1)/W;       bh = (y2-y1)/H
                yolo_lines.append(f"{arrow_cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")


        # Aumentações globais (moderadas)
        if random.random() < 0.7:
            img = img.rotate(random.uniform(-args.skew_max_deg, args.skew_max_deg),
                             resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255,0))
        if random.random() < args.p_perspective:
            img = perspective_warp(img, max_warp=args.persp_max)
        if random.random() < args.p_downup:
            s = random.uniform(0.6, 0.9)
            img = img.resize((int(W*s), int(H*s)), Image.BICUBIC).resize((W,H), Image.BICUBIC)
        if random.random() < 0.7:
            if random.random() < args.p_motion_blur:
                k = random.choice([3,5,7,9])  # seguro (ímpar)
                img = motion_blur_pil(img, k=k)
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, args.blur_sigma)))
        if random.random() < args.p_noise:
            img = add_noise(img, std=args.noise_std)
        if random.random() < args.p_jpeg:
            img = jpeg_compress(img, q_min=args.jpeg_min, q_max=args.jpeg_max)

        # salva
        img_path = out / "images" / f"syn_{i:06d}.jpg"
        img.convert("RGB").save(img_path, quality=95)
        img_paths.append(str(img_path))

        # labels YOLO
        yolo_lines = []
        for (x1,y1,x2,y2,cls) in boxes:
            cid = name_to_id[cls]
            xc = ((x1+x2)/2.0)/W
            yc = ((y1+y2)/2.0)/H
            bw = (x2-x1)/W
            bh = (y2-y1)/H
            yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(out / "labels" / f"syn_{i:06d}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

        if args.emit_graph:
            # Calcula boundaries para esta imagem
            boundaries = []
            for j, r in enumerate(panel_rects):
                b_nodes = [n["id"] for n in nodes if inside(r, n["bbox"])]
                boundaries.append({"id": j, "bbox": list(r), "nodes": b_nodes})
        
            graph = {
                "image": str(img_path.name),
                "size": [W, H],
                "nodes": nodes,
                "edges": edges,
                "boundaries": boundaries,
                "ocr": ocr_items
            }
            with open(out / "graphs" / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)

    # classes.yaml e data.yaml
    with open(out / "classes.yaml", "w") as f:
        yaml.safe_dump({"names": classes}, f, sort_keys=False, allow_unicode=True)

    # splits (txt com paths absolutos de imagens)
    random.shuffle(img_paths)
    n_val = max(1, int(len(img_paths) * args.val_ratio))
    val = sorted(img_paths[:n_val])
    train = sorted(img_paths[n_val:])
    (out / "splits").mkdir(parents=True, exist_ok=True)
    with open(out / "splits" / "train.txt", "w") as f: f.write("\n".join(train))
    with open(out / "splits" / "val.txt", "w") as f: f.write("\n".join(val))

    # yolo data.yaml (multi-source via splits)
    data_yaml = {
        "path": str(out.resolve()),
        "train": [str((out/"splits"/"train.txt").resolve())],
        "val":   [str((out/"splits"/"val.txt").resolve())],
        "names": classes
    }
    with open(out / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print("\n[OK] Dataset sintético gerado:")
    print(" - imagens:", out / "images")
    print(" - labels :", out / "labels")
    print(" - data.yaml:", out / "data.yaml")
    print(" - splits  :", out / "splits" / "train.txt", "e", out / "splits" / "val.txt")

if __name__ == "__main__":
    main()
