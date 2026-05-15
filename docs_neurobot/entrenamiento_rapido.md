# Guía de Entrenamiento Rápido — NeuroPolicy

Pipeline completo para entrenar los modelos que necesita `NeuroPolicy` y dejarlos listos para el build Docker.

---

## Resumen de lo que hay que entrenar

| Modelo | Script | Destino final | Tiempo estimado |
|---|---|---|---|
| Keypoints SC | `train_keypoint_estimator.py` | `models/keypoints/best_sc.pt` | ~30 min |
| Keypoints SFP | `train_keypoint_estimator.py` | `models/keypoints/best_sfp.pt` | ~30 min |
| ACT policy | `train_act_fast.py` | `models/act_policy/` | **10–15 min** |
| Calibración F/T | `calibrate_ft.py` | `bayesiano/calibration_results.csv` | opcional |

Todo se ejecuta desde WSL (`/mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic`).

---

## Paso 0 — Prerequisitos

```bash
# Desde WSL, en la raíz del repo
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic

# Verificar que pixi está disponible
pixi --version
```

Los keypoints y la calibración F/T necesitan los bags de Dataset A y Dataset B respectivamente. Consultar [training_architecture.md](training_architecture.md) para generarlos.

---

## Paso 1 — Entrenar Keypoints

El estimador de keypoints usa MobileNetV3-Small y predice 9 puntos en imagen (u, v) normalizados a [0,1].

### Entrenamiento SC

```bash
pixi run python scripts/train_keypoint_estimator.py \
  --dataset_dir ~/aic_datasets/dataset_A \
  --connector_type sc \
  --epochs 50 \
  --batch_size 32 \
  --output_dir fine_tune_act_v2/keypoints/keypoint_sc_1h
```

### Entrenamiento SFP

```bash
pixi run python scripts/train_keypoint_estimator.py \
  --dataset_dir ~/aic_datasets/dataset_A \
  --connector_type sfp \
  --epochs 50 \
  --batch_size 32 \
  --output_dir fine_tune_act_v2/keypoints/keypoint_sfp_1h
```

Los checkpoints se guardan como `best_sc.pt` y `best_sfp.pt` dentro de cada `output_dir`.

### Copiar checkpoints al repo

```bash
mkdir -p models/keypoints

cp fine_tune_act_v2/keypoints/keypoint_sc_1h/best_sc.pt   models/keypoints/best_sc.pt
cp fine_tune_act_v2/keypoints/keypoint_sfp_1h/best_sfp.pt models/keypoints/best_sfp.pt
```

---

## Paso 2 — Convertir bags a LeRobotDataset

Los bags ROS 2 del Dataset B (CheatCode) o Dataset C (teleop) se convierten al formato que `train_act_fast.py` espera.

```bash
cd fine_tune_act_v2/fine_tune_act

# Dataset B — CheatCode (o Dataset C, usar --append para agregar)
pixi run python scripts/bag_to_lerobot.py \
  --bags_dir ~/aic_bags/dataset_b \
  --output   data/dataset_lerobot \
  --repo_id  aic_team/sfp_sc_insertion \
  --ckpt_sc  ../../models/keypoints/best_sc.pt \
  --ckpt_sfp ../../models/keypoints/best_sfp.pt \
  --fps 10
```

Para agregar Dataset C (teleop) al mismo dataset:

```bash
pixi run python scripts/bag_to_lerobot.py \
  --bags_dir ~/aic_bags/dataset_c \
  --output   data/dataset_lerobot \
  --repo_id  aic_team/sfp_sc_insertion \
  --ckpt_sc  ../../models/keypoints/best_sc.pt \
  --ckpt_sfp ../../models/keypoints/best_sfp.pt \
  --fps 10 \
  --append
```

> `bag_to_lerobot.py` detecta automáticamente el tipo de conector (sfp/sc) desde `/tf` o desde el nombre del bag.

**Verificar el dataset:**

```bash
pixi run python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id='aic_team/sfp_sc_insertion', root='data/dataset_lerobot')
print(f'Episodios: {ds.num_episodes}')
print(f'Frames:    {len(ds)}')
print(f'FPS:       {ds.fps}')
"
```

---

## Paso 3 — Entrenar ACT (modo rápido)

### Modo por defecto — 10 000 steps, imágenes 240×320, ~10-15 min

```bash
cd fine_tune_act_v2/fine_tune_act

pixi run python scripts/train_act_fast.py
```

Este modo:
- Congela los 3 backbones ResNet18 (no hay backward por ellos → ~50% menos tiempo)
- Reduce imágenes de 480×640 a 240×320 en GPU (~4× forward más rápido)
- Solo entrena: transformer + state_proj + action_head

### Opciones útiles

```bash
# Más steps si se quiere mejor convergencia (~20-25 min)
pixi run python scripts/train_act_fast.py --steps 25000

# Resolución original 480×640 (backbone congelado, sin resize, ~1.5-2h)
pixi run python scripts/train_act_fast.py \
  --steps 25000 \
  --img_height 480 --img_width 640

# Continuar desde un checkpoint existente
pixi run python scripts/train_act_fast.py \
  --ckpt outputs/act_fast_run1/checkpoints/step_010000
```

### Log esperado

```
=== pixi install pass ... ===
TRAIN ACT FAST — backbone congelado + imágenes reducidas
  device      : cuda
  steps       : 10000
  img_size    : 240×320 [REDUCIDO — 4× más rápido]
  ...
  step    100/10000  loss=0.0842  lr=1.67e-05  0.1min / ETA 12.3min
  step    200/10000  loss=0.0631  lr=3.33e-05  0.2min / ETA 11.8min
  ...
  [ckpt] outputs/act_fast_run1/checkpoints/step_002000
  ...
  [ckpt] outputs/act_fast_run1/checkpoints/step_010000
Entrenamiento completado en 12.4 min
```

### Arquitectura ACT (de `act_aic.yaml`)

| Parámetro | Valor |
|---|---|
| Vision backbone | ResNet18 × 3 cámaras |
| Transformer dim | 512 |
| Heads | 8 |
| Encoder layers | 4 |
| chunk_size | 20 |
| n_action_steps | 20 |
| State input | 32D |
| Action output | 6D twist |
| VAE latent dim | 32 |

---

## Paso 4 — Copiar pesos ACT al repo

```bash
# Volver a la raíz del repo
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic

mkdir -p models/act_policy

cp -r fine_tune_act_v2/fine_tune_act/outputs/act_fast_run1/checkpoints/step_010000/* \
      models/act_policy/
```

Verificar que existen los dos archivos necesarios:

```bash
ls -lh models/act_policy/
# config.json         ← configuración de ACTPolicy
# model.safetensors   ← pesos del modelo
```

> Si se entrenó con más steps, reemplazar `step_010000` por el número correspondiente (ej: `step_025000`).

---

## Paso 5 — Calibración F/T (opcional)

Si se tiene el bag de calibración del Axia80:

```bash
pixi run python tools/calibrate_ft.py \
  --calib ~/aic_bags/ft_calib_TIMESTAMP

# Genera bayesiano/calibration_results.csv
ls -lh bayesiano/calibration_results.csv
```

Si no se tiene el bag, `NeuroPolicy` usa los valores por defecto hardcodeados en `bayesian_estimator.py` — el policy funciona igual pero sin calibración personalizada del sensor.

---

## Paso 6 — Build Docker con los modelos reales

```bash
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic

docker build -f docker/aic_model/Dockerfile -t aic_model:latest .
```

El Dockerfile copia `models/` y `bayesiano/` dentro de la imagen en `/opt/aic/`.

### Verificar que los modelos están dentro de la imagen

```bash
docker run --rm aic_model:latest ls -lh /opt/aic/models/act_policy/
# config.json
# model.safetensors

docker run --rm aic_model:latest ls -lh /opt/aic/models/keypoints/
# best_sc.pt
# best_sfp.pt
```

---

## Paso 7 — Prueba local antes de subir

```bash
# Necesita DISPLAY configurado (WSL con X server o ejecutar en Linux nativo)
xhost +local:docker

docker compose -f docker/docker-compose.yaml up
```

---

## Resumen de archivos generados

```
fine_tune_act_v2/fine_tune_act/
  outputs/act_fast_run1/checkpoints/step_010000/
    config.json
    model.safetensors

fine_tune_act_v2/keypoints/
  keypoint_sc_1h/best_sc.pt
  keypoint_sfp_1h/best_sfp.pt

models/                          ← copiados aquí para el Docker build
  act_policy/
    config.json
    model.safetensors
  keypoints/
    best_sc.pt
    best_sfp.pt

bayesiano/                       ← opcional
  calibration_results.csv
```

---

## Nota sobre resolución de imagen

`train_act_fast.py` por defecto entrena con 240×320. El Dockerfile tiene:

```
ENV AIC_IMG_H=240
ENV AIC_IMG_W=320
```

Si se entrena con resolución original (`--img_height 480 --img_width 640`), cambiar esas variables en [docker/aic_model/Dockerfile](../docker/aic_model/Dockerfile) antes del build:

```dockerfile
ENV AIC_IMG_H=480
ENV AIC_IMG_W=640
```
