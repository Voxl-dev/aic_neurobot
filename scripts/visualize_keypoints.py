r"""
Visualiza los keypoints de una muestra del dataset A.

Uso:
  python visualize_keypoints.py <ruta_dataset_A> [sample_index]
  python visualize_keypoints.py ~/aic_datasets/dataset_A 18 --save /tmp/sample18.png

Notas:
  - El índice es la posición dentro de los .json ordenados, no el sample_id interno.
  - Por defecto aplica auto-contraste al recorte porque las primeras muestras
    del dataset pueden verse muy oscuras.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualiza keypoints del dataset A.")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=".",
        help="Ruta al directorio del dataset A.",
    )
    parser.add_argument(
        "sample_index",
        nargs="?",
        type=int,
        default=0,
        help="Indice dentro de la lista ordenada de archivos JSON.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=80,
        help="Margen en pixeles alrededor del bounding box de keypoints visibles.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Escala final del mosaico mostrado/guardado.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Si se pasa, guarda el mosaico anotado en esta ruta.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abre ventana; util si solo quieres guardar el PNG.",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Desactiva el auto-contraste del recorte.",
    )
    return parser.parse_args()


def auto_contrast(img: np.ndarray) -> np.ndarray:
    """Estira el rango dinamico del recorte para que puertos oscuros se vean."""
    lo = float(np.percentile(img, 1.0))
    hi = float(np.percentile(img, 99.5))
    if hi <= lo:
        lo = float(img.min())
        hi = float(img.max())
    if hi <= lo:
        return img
    stretched = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()

    json_files = sorted(dataset_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"No se encontraron .json en {dataset_dir}")

    if args.sample_index < 0 or args.sample_index >= len(json_files):
        sys.exit(
            f"sample_index fuera de rango: {args.sample_index}. "
            f"Hay {len(json_files)} muestras."
        )

    sample_file = json_files[args.sample_index]
    data = json.loads(sample_file.read_text())
    print(
        f"Cargando indice {args.sample_index}: {sample_file.name}  |  "
        f"sample_id: {data['sample_id']}  |  conector: {data['connector_type']}  |  "
        f"frame: {data['port_frame']}"
    )

    colors = {"left": (0, 255, 0), "center": (0, 200, 255), "right": (255, 80, 80)}
    panels = []

    for cam in ("left", "center", "right"):
        img_path = dataset_dir / data["images"][cam]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [!] No se pudo leer {img_path}")
            continue

        kps = data[f"keypoints_{cam}"]
        color = colors[cam]
        H, W = img.shape[:2]

        # Filtrar keypoints visibles (dentro de los limites de la imagen).
        visible = [(x, y) for x, y in kps if 0 <= x < W and 0 <= y < H]
        if not visible:
            print(f"  [~] {cam}: puerto fuera del campo de vision ({len(kps)} kps fuera de imagen)")
            continue

        if len(visible) < len(kps):
            print(
                f"  [~] {cam}: {len(kps) - len(visible)} kps fuera de imagen, "
                f"{len(visible)} visibles"
            )

        # Recortar alrededor del puerto usando la bounding box de los keypoints visibles.
        xs = [x for x, y in visible]
        ys = [y for x, y in visible]
        x1 = max(0, int(min(xs)) - args.padding)
        y1 = max(0, int(min(ys)) - args.padding)
        x2 = min(W, int(max(xs)) + args.padding)
        y2 = min(H, int(max(ys)) + args.padding)
        img = img[y1:y2, x1:x2].copy()

        if not args.no_enhance:
            img = auto_contrast(img)

        for i, (x, y) in enumerate(kps):
            # Dibujar solo los keypoints visibles.
            if not (0 <= x < W and 0 <= y < H):
                continue
            pt = (int(round(x)) - x1, int(round(y)) - y1)
            cv2.circle(img, pt, 5, color, -1)
            cv2.putText(
                img,
                str(i),
                (pt[0] + 4, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        cv2.putText(img, cam, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        panels.append(img)

    if not panels:
        sys.exit("No se pudo cargar ninguna imagen.")

    # Escalar todas al mismo alto antes de concatenar.
    h = min(p.shape[0] for p in panels)
    resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    mosaic = np.concatenate(resized, axis=1)

    if args.scale != 1.0:
        mosaic = cv2.resize(
            mosaic,
            (int(mosaic.shape[1] * args.scale), int(mosaic.shape[0] * args.scale)),
            interpolation=cv2.INTER_NEAREST,
        )

    if args.save:
        save_path = args.save.expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), mosaic):
            sys.exit(f"No se pudo guardar la imagen anotada en {save_path}")
        print(f"Mosaico guardado en: {save_path}")

    if args.no_show:
        return 0

    title = f"sample {data['sample_id']} - {data['connector_type']} - {Path(sample_file.name).stem}"
    cv2.imshow(title, mosaic)
    print("Pulsa cualquier tecla para cerrar.")
    try:
        cv2.waitKey(0)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
