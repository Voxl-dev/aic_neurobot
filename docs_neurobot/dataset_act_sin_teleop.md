# Dataset ACT sin teleop — generación en < 1 hora

**Hardware objetivo:** RTX 2000 Ada 8GB · 31GB RAM · Linux nativo · CUDA 12.x/13.x

Sin teleop (Dataset C), el ACT se entrena **solo con CheatCode en Gazebo**.
60 episodios son suficientes para un ACT funcional en la tarea de inserción.
El proceso completo (grabar + convertir) toma ~45 minutos en este hardware.

---

## Mapa de tiempo

| Fase | Duración | Qué hace |
|---|---|---|
| Startup Docker + Gazebo | 2 min (una sola vez) | Contenedor eval levanta la simulación |
| Grabación 60 episodios | ~22 min | Loop: CheatCode resuelve el trial, se graba un bag |
| Conversión bag_to_lerobot | ~12 min | GPU infiere keypoints, escribe LeRobotDataset |
| Verificación | 2 min | Checks de integridad del dataset |
| **Total** | **~38 min** | |

El entrenamiento ACT (~5–6h) corre por separado en GPU después.

---

## Prerequisitos

```bash
# Desde el directorio del repo (en Linux nativo)
cd ~/path/to/aic

# Dar acceso de pantalla al contenedor (para Gazebo headless no es estrictamente necesario)
xhost +local:docker

# Variables de entorno para el loop de grabación
mkdir -p data/bags_cheatcode
```

Verificar que Docker ve la GPU:
```bash
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi | head -5
```

---

## Paso 1 — Grabar 60 episodios con CheatCode

El truco de velocidad: **mantener Gazebo activo** entre episodios y solo reiniciar
el nodo de política + la grabación del bag. Se evitan los ~60 segundos de startup
de Gazebo por episodio.

### Terminal A — Simulación persistente

```bash
docker run --rm \
  --gpus all \
  --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e AIC_EVAL_PASSWD=train_local \
  -e AIC_MODEL_PASSWD=train_local \
  ghcr.io/intrinsic-dev/aic/aic_eval \
  gazebo_gui:=false \
  ground_truth:=true \
  start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=false \
  model_discovery_timeout_seconds:=7200
```

Esperar hasta ver en los logs:
```
[gz_ros2_control]: Loaded state interface: shoulder_pan_joint/position
[aic_engine]: Waiting for model discovery...
```
Esto tarda ~60 segundos la primera vez.

### Terminal B — Loop de grabación (60 episodios)

Ejecutar en el **host Linux** (no dentro del contenedor), con acceso al namespace
de red del contenedor via `--network host`:

```bash
#!/usr/bin/env bash
# Guardar como scripts/record_act_dataset.sh y ejecutar con bash
set -euo pipefail

N_EPISODES=60
BAG_DIR="data/bags_cheatcode"
mkdir -p "$BAG_DIR"

# Conectar al router Zenoh del contenedor eval
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_ROUTER_CHECK_ATTEMPTS=3
export AIC_ROUTER_ADDR=localhost:7447

for i in $(seq -w 1 $N_EPISODES); do
    echo "══════════ Episodio $i / $N_EPISODES ══════════"
    BAG_PATH="${BAG_DIR}/episode_${i}"

    # 1. Iniciar grabación en segundo plano
    ros2 bag record \
        /center_camera/image \
        /left_camera/image \
        /right_camera/image \
        /joint_states \
        /tf \
        /tf_static \
        --max-bag-duration 90 \
        -o "$BAG_PATH" &
    BAG_PID=$!
    sleep 1   # Darle tiempo al recorder para suscribirse

    # 2. Ejecutar un trial con CheatCode (bloquea hasta que el trial termina)
    ros2 run aic_model aic_model \
        --ros-args \
        -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.CheatCode \
        2>/dev/null || true

    # 3. Detener grabación
    kill "$BAG_PID" 2>/dev/null
    wait "$BAG_PID" 2>/dev/null || true

    # Verificación rápida del bag
    FRAMES=$(ros2 bag info "$BAG_PATH" 2>/dev/null | grep center_camera | grep -oP '\d+ msgs' | head -1 || echo "?")
    echo "  Bag guardado: $BAG_PATH  ($FRAMES frames center)"
    sleep 2   # Esperar reset de la simulación
done

echo "Grabación completa: $N_EPISODES episodios en $BAG_DIR"
```

```bash
chmod +x scripts/record_act_dataset.sh
bash scripts/record_act_dataset.sh
```

**Qué esperar:**
- Cada episodio tarda ~20 segundos (CheatCode es determinista y rápido)
- Los bags quedan en `data/bags_cheatcode/episode_001/` ... `episode_060/`
- Cada bag pesa ~150-300MB según la duración del episodio
- Total en disco: ~15GB para 60 episodios

### Si el trial no termina solo

Algunos builds del engine requieren `shutdown_on_aic_engine_exit:=false`.
Si el nodo `aic_model` no finaliza después de ~60 segundos, agregar timeout:

```bash
timeout 60 ros2 run aic_model aic_model \
    --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.CheatCode \
    2>/dev/null || true
```

---

## Paso 2 — Convertir bags a LeRobotDataset

Con la GPU, `bag_to_lerobot.py` infiere los 9 keypoints por frame en ~5ms.
60 episodios × ~200 frames/ep = 12,000 frames → ~60 segundos de inferencia sola.

```bash
pixi run python fine_tune_act_v2/fine_tune_act/scripts/bag_to_lerobot.py \
    --bags_dir   data/bags_cheatcode \
    --output     data/dataset_lerobot \
    --repo_id    aic_team/sfp_sc_insertion \
    --ckpt_sc    fine_tune_act_v2/keypoints/keypoint_sc_1h/best_sc.pt \
    --ckpt_sfp   fine_tune_act_v2/keypoints/keypoint_sfp_1h/best_sfp.pt \
    --fps        10 \
    --device     cuda \
    --connector_type auto
```

**Flags importantes para este hardware:**

| Flag | Valor | Razón |
|---|---|---|
| `--fps 10` | 10 Hz | Reduce tamaño sin perder dinámica de inserción |
| `--device cuda` | cuda | RTX 2000 Ada: 10× más rápido que CPU |
| `--connector_type auto` | auto | Detecta sfp/sc desde `/tf` automáticamente |

**Salida esperada:**
```
Cargando modelos de keypoints ...
  SC  → best_sc.pt   device=cuda
  SFP → best_sfp.pt  device=cuda

Bags encontrados: 60  (en data/bags_cheatcode)

[1/60] episode_001
  Indexando episode_001 ...
  OK  187 frames  conector=sfp
  episodio 1 guardado  (elapsed=0.4min  ETA=23.1min)
...
[60/60] episode_060
  ...

Consolidando dataset (58 episodios, 2 skipped) ...
════════════════════════════════════════════
  COMPLETADO — 58 episodios en 11.2 min
  Dataset: data/dataset_lerobot
```

2 skips son normales (bags corruptos o demasiado cortos).

---

## Paso 3 — Verificar el dataset

```bash
# Número de episodios y frames totales
pixi run python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id='aic_team/sfp_sc_insertion', root='data/dataset_lerobot')
print(f'Episodios : {ds.num_episodes}')
print(f'Frames    : {len(ds)}')
print(f'FPS       : {ds.fps}')
print(f'Features  : {list(ds.features.keys())}')
"
```

Salida esperada:
```
Episodios : 58
Frames    : ~11000
FPS       : 10
Features  : ['observation.images.center', 'observation.images.left',
             'observation.images.right', 'observation.state', 'action']
```

```bash
# Verificar shape del state [32] y action [6]
pixi run python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
ds = LeRobotDataset(repo_id='aic_team/sfp_sc_insertion', root='data/dataset_lerobot')
sample = ds[0]
print('state shape :', sample['observation.state'].shape)   # torch.Size([32])
print('action shape:', sample['action'].shape)               # torch.Size([6])
img = sample['observation.images.center']
print('center shape:', img.shape)                            # torch.Size([3, 480, 640])
"
```

```bash
# Distribución de acciones (no deben tener NaN ni valores extremos)
pixi run python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
ds = LeRobotDataset(repo_id='aic_team/sfp_sc_insertion', root='data/dataset_lerobot')
actions = torch.stack([ds[i]['action'] for i in range(min(500, len(ds)))])
print('action mean :', actions.mean(0).numpy().round(4))
print('action std  :', actions.std(0).numpy().round(4))
print('action max  :', actions.abs().max(0).values.numpy().round(3))
print('NaN count   :', actions.isnan().sum().item())
"
```

Valores típicos para Cartesian twist en mm/s:
- `vx, vy` std: ~0.01–0.05 m/s
- `vz` std: ~0.01–0.03 m/s (movimiento principal de inserción)
- max absoluto: < 0.2 m/s (CheatCode es suave)

---

## Paso 4 — Lanzar entrenamiento ACT

Con el dataset listo, el entrenamiento corre directamente:

```bash
bash fine_tune_act_v2/fine_tune_act/scripts/run_finetune.sh
```

**Duración en RTX 2000 Ada 8GB:**

| Fase | Tiempo |
|---|---|
| `prepare_checkpoint.py` (descarga + smart init) | ~3 min |
| `lerobot-train` 60k steps, batch=8, AMP | ~4.5–5.5h |
| **Total entrenamiento** | **~5h** |

El script guarda checkpoints cada 5000 steps (~25 min). Si hay corte, el entrenamiento
puede resumirse desde el último checkpoint.

**Monitoreo de la pérdida durante entrenamiento:**
```bash
# En otra terminal mientras entrena
tail -f outputs/act_aic_run1/.hydra/hydra.log | grep "loss"
```

Valores esperados con 58 episodios:
- Step 1k:  loss ~ 0.35–0.50 (warmup)
- Step 10k: loss ~ 0.08–0.15
- Step 60k: loss ~ 0.02–0.06

---

## Si el dataset queda con < 40 episodios

Con menos de 40 episodios el ACT puede overfit. Opciones:

**A) Reducir learning rate** en `fine_tune_act_v2/fine_tune_act/configs/act_aic.yaml`:
```yaml
training:
  lr: 5.0e-6          # Cambiar de 1.0e-5 a 5.0e-6
  offline_steps: 30000 # Reducir steps proporcional
```

**B) Aumentar el dataset con los mismos bags:**
```bash
pixi run python fine_tune_act_v2/fine_tune_act/scripts/bag_to_lerobot.py \
    --bags_dir   data/bags_cheatcode \
    --output     data/dataset_lerobot \
    --repo_id    aic_team/sfp_sc_insertion \
    --ckpt_sc    fine_tune_act_v2/keypoints/keypoint_sc_1h/best_sc.pt \
    --ckpt_sfp   fine_tune_act_v2/keypoints/keypoint_sfp_1h/best_sfp.pt \
    --fps        15 \     # Más frames por episodio (15 Hz vs 10 Hz)
    --device     cuda \
    --append
```

**C) Grabar más episodios** (~20 más = 15 minutos extra de simulación):
```bash
# Reutilizar el mismo script, el --append agrega al dataset existente
N_EPISODES=20 bash scripts/record_act_dataset.sh
pixi run python fine_tune_act_v2/fine_tune_act/scripts/bag_to_lerobot.py \
    --bags_dir data/bags_cheatcode \
    --output   data/dataset_lerobot \
    --repo_id  aic_team/sfp_sc_insertion \
    --ckpt_sc  fine_tune_act_v2/keypoints/keypoint_sc_1h/best_sc.pt \
    --ckpt_sfp fine_tune_act_v2/keypoints/keypoint_sfp_1h/best_sfp.pt \
    --fps 10 --device cuda --append
```

---

## Checklist de salida

Al terminar el Paso 3, verificar:

- [ ] `ds.num_episodes >= 40`
- [ ] `sample['observation.state'].shape == torch.Size([32])`
- [ ] `sample['action'].shape == torch.Size([6])`
- [ ] `sample['observation.images.center'].shape == torch.Size([3, 480, 640])`
- [ ] `actions.isnan().sum() == 0`
- [ ] `actions.abs().max() < 1.0` (sin velocidades explosivas)
- [ ] Directorio `data/dataset_lerobot/videos/` existe y tiene archivos `.mp4`
