r"""
Visualiza los keypoints de una muestra del dataset A.
Uso:
  python visualize_keypoints.py <ruta_dataset_A> [sample_id]
Ejemplo (Windows):
  python visualize_keypoints.py "\\wsl$\Ubuntu\home\diego\aic_datasets\dataset_A" 0
"""
import sys
import json
import glob
import cv2
import numpy as np
from pathlib import Path

dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
sample_id   = int(sys.argv[2]) if len(sys.argv) > 2 else 0

json_files = sorted(dataset_dir.glob("*.json"))
if not json_files:
    sys.exit(f"No se encontraron .json en {dataset_dir}")

sample_file = json_files[sample_id]
data = json.loads(sample_file.read_text())
print(f"Cargando: {sample_file.name}  |  conector: {data['connector_type']}  |  frame: {data['port_frame']}")

COLORS = {"left": (0, 255, 0), "center": (0, 200, 255), "right": (255, 80, 80)}

panels = []
for cam in ("left", "center", "right"):
    img_path = dataset_dir / data["images"][cam]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [!] No se pudo leer {img_path}")
        continue

    kps = data[f"keypoints_{cam}"]
    color = COLORS[cam]
    H, W = img.shape[:2]

    # Filtrar keypoints visibles (dentro de los límites de la imagen)
    visible = [(x, y) for x, y in kps if 0 <= x < W and 0 <= y < H]
    if not visible:
        print(f"  [~] {cam}: puerto fuera del campo de visión ({len(kps)} kps fuera de imagen)")
        continue

    if len(visible) < len(kps):
        print(f"  [~] {cam}: {len(kps) - len(visible)} kps fuera de imagen, "
              f"{len(visible)} visibles")

    # Recortar alrededor del puerto usando la bounding box de los keypoints visibles
    PADDING = 80  # píxeles de margen alrededor del puerto
    xs = [x for x, y in visible]
    ys = [y for x, y in visible]
    x1 = max(0, int(min(xs)) - PADDING)
    y1 = max(0, int(min(ys)) - PADDING)
    x2 = min(W, int(max(xs)) + PADDING)
    y2 = min(H, int(max(ys)) + PADDING)
    img = img[y1:y2, x1:x2].copy()

    for i, (x, y) in enumerate(kps):
        # Dibujar sólo los keypoints visibles
        if not (0 <= x < W and 0 <= y < H):
            continue
        pt = (int(round(x)) - x1, int(round(y)) - y1)
        cv2.circle(img, pt, 5, color, -1)
        cv2.putText(img, str(i), (pt[0]+4, pt[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(img, cam, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    panels.append(img)

if not panels:
    sys.exit("No se pudo cargar ninguna imagen.")

# Escalar todas al mismo alto antes de concatenar
h = min(p.shape[0] for p in panels)
resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
mosaic = np.concatenate(resized, axis=1)

title = f"sample {data['sample_id']} — {data['connector_type']} — {Path(sample_file.name).stem}"
cv2.imshow(title, mosaic)
print("Pulsa cualquier tecla para cerrar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
