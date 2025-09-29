import os
import time
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento YOLO com suporte MPS")
    parser.add_argument('--data', type=str, required=True, help='Caminho para o data.yaml')
    parser.add_argument('--runs_dir', type=str, default='runs', help='Diretório dos resultados')
    parser.add_argument('--run_name', type=str, default='yolo_syn_real_mps_s', help='Nome do experimento')
    parser.add_argument('--base_model', type=str, default='yolov8s.pt', help='Modelo base YOLO')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamanho da imagem')
    parser.add_argument('--epochs', type=int, default=60, help='Número de épocas')
    parser.add_argument('--batch', type=int, default=12, help='Tamanho do batch')
    parser.add_argument('--device', type=str, default='mps', help='Dispositivo (ex: mps, cpu, cuda)')
    parser.add_argument('--workers', type=int, default=0, help='Número de workers')
    parser.add_argument('--resume', action='store_true', help='Retomar do último checkpoint se existir')
    return parser.parse_args()

def main():
    args = parse_args()
    last_pt = f"{args.runs_dir}/detect/{args.run_name}/weights/last.pt"
    train_cfg = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=False,
        mosaic=0, mixup=0, copy_paste=0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=3, translate=0.06, scale=0.9, perspective=0.0,
        cache='disk',
        val=False,
        project=args.runs_dir,
        name=args.run_name,
        exist_ok=True,
        resume=args.resume or os.path.exists(last_pt),
        save_period=1,
        plots=False
    )
    retries = 5
    for attempt in range(retries):
        try:
            model_path = last_pt if os.path.exists(last_pt) else args.base_model
            model = YOLO(model_path)
            model.train(**train_cfg)
            print("[OK] treino finalizado.")
            return
        except KeyboardInterrupt:
            print("[INFO] Interrompido. Você pode rodar de novo para retomar do last.pt.")
            return
        except Exception as e:
            print(f"[MPS][tentativa {attempt+1}] crash: {e}")
            time.sleep(10)
    print("[ERRO] excedeu tentativas. Verifique logs do último run em:", f"{args.runs_dir}/detect/{args.run_name}")

if __name__ == "__main__":
    main()