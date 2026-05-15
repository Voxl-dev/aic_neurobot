# Arquitectura de Entrenamiento y Datasets — AIC Qualification Phase

## Visión general

Este documento describe la arquitectura de solución propuesta, los datasets
necesarios para entrenarla y los comandos exactos para generarlos y verificarlos.
Está alineado con las reglas del reto (`challenge_rules.md`), las fases
(`qualification_phase.md`, `phases.md`), el sistema de scoring (`scoring.md`) y
la infraestructura disponible en el repositorio.

---

## 1. Arquitectura del sistema propuesto

### 1.1 Diagrama de flujo (inferencia en tiempo real)

```
Gazebo / Robot real (20 Hz)
  │
  ├─ 3 × RGB Image (1152×1024) ──────────────────────┐
  ├─ /aic_controller/controller_state (TCP pose/vel) ─┤
  ├─ /fts_broadcaster/wrench (F/T 6D) ────────────────┤
  └─ /joint_states (6 joints) ───────────────────────┘
                                                       │
                                         ┌─────────────▼──────────────┐
                                         │   CAPA 0 — Keypoint        │
                                         │   Pose Estimator           │
                                         │   MobileNetV3-Small × 3    │
                                         │   → pose_relativa [6D]     │
                                         │   (dx,dy,dz,dRx,dRy,dRz)  │
                                         └─────────────┬──────────────┘
                                                       │
                                         ┌─────────────▼──────────────┐
                                         │   Vector de estado [32D]   │
                                         │   tcp_pose       [7]       │
                                         │   tcp_velocity   [6]       │
                                         │   joint_pos      [7]       │
                                         │   ft_tared       [6]       │
                                         │   keypoint_pose  [6]       │
                                         └─────────────┬──────────────┘
                                                       │
                                         ┌─────────────▼──────────────┐
                                         │   CAPA 2 — ACT             │
                                         │   ResNet-18 (ImageNet)     │
                                         │   Transformer dim=512      │
                                         │   chunk_size=20            │
                                         │   → action Twist [6D]      │
                                         └─────────────┬──────────────┘
                                                       │
                       ┌───────────────────────────────┤
                       │  dist_plug_tip ≤ 5mm?         │
                       │  (estimado por Capa 0)        │
                       └─────────┬─────────────────────┘
                                 │ Sí
                   ┌─────────────▼──────────────┐
                   │   CAPA 3 — SAC residual     │  (entrenado offline,
                   │   MLP 256×256              │   activo solo en
                   │   → Δaction [3D]           │   los últimos 5mm)
                   └─────────────┬──────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │   FSM — Safety Monitor     │  (50 Hz, F/T check)
                   │   F/T > 20N por >1s → RETRACT
                   │   Reintentos ≤ 3           │
                   │   Timeout 160s             │
                   └─────────────┬──────────────┘
                                 │
                /aic_controller/pose_commands (MODE_VELOCITY)
                                 │
                   Impedancia cartesiana → UR5e
```

### 1.2 Diferencias clave respecto al RunACT de referencia

| Aspecto                     | `RunACT` de referencia              | Esta propuesta                                  |
| --------------------------- | ------------------------------------- | ----------------------------------------------- |
| Localización del puerto    | Implícita — ACT aprende desde 0     | Explícita — Capa 0 da pose[6D] directa        |
| Vector de estado            | 26D (sin F/T en state, sin keypoints) | 32D (con F/T tared + keypoints)                 |
| chunk_size                  | 100 acciones (~25s horizon)           | 20 acciones (~2s), más reactivo a F/T          |
| Control en los últimos 5mm | ACT sin señal especializada          | SAC residual con reward de inserción           |
| Protección F/T             | Ninguna en código                    | FSM a 50Hz, evita -12/-24 pts de penalty        |
| Fuente de datos             | Checkpoint HF descargado              | Dataset propio desde Gazebo + teleop            |
| Entrenamiento en otros sims | No aplica                             | Isaac Lab disponible en `aic_utils/aic_isaac` |

### 1.3 Relación con lo ya disponible en el repo

| Archivo del repo                          | Rol en la propuesta                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------- |
| `aic_example_policies/ros/CheatCode.py` | Generador automático de Dataset B                                                    |
| `aic_example_policies/ros/RunACT.py`    | Plantilla de `insert_cable()` para Capa 2                                           |
| `aic_utils/lerobot_robot_aic/`          | `lerobot-record` y `lerobot-train` para Dataset C                                 |
| `aic_utils/aic_teleoperation/`          | Teleoperation alternativa (sin LeRobot) para Dataset C                                |
| `aic_utils/aic_isaac/aic_isaaclab/`     | Isaac Lab con `randomize_dome_light` y `randomize_board_and_parts` para Dataset A |
| `aic_utils/aic_training_utils/`         | `aic_training_gz_bringup.launch.py` + `xacro_expander` para spawning automático  |
| `aic_engine/config/sample_config.yaml`  | Rangos de randomización del task board                                               |
| `aic_bringup/`                          | Launch files para distintas configuraciones de simulación                            |

---

## 2. Datasets requeridos

### Dataset A — Keypoint Pose Estimator (sintético, automático)

**Para qué:** Entrenar la Capa 0 (MobileNetV3). El estimador produce la pose
relativa del puerto respecto al TCP, que es el feature de percepción explícita
que el ACT recibe como entrada.

**Fuente:** Gazebo local con `ground_truth:=true`. Los frames `/tf` dan la pose
exacta de cada puerto. Proyección geométrica automática → labels de keypoints
en imagen. Cero anotación manual.

**Reglas del reto:** Permitido. `challenge_rules.md` sección 2.c:

> "During training, participants may use all internal state information, including
> ground truth data available over the /tf topic."

**Estructura por muestra:**

```json
{
  "connector_type": "sfp | sc",
  "keypoints_left":   [[u1,v1], "...", [u8,v8]],
  "keypoints_center": [[u1,v1], "...", [u8,v8]],
  "keypoints_right":  [[u1,v1], "...", [u8,v8]],
  "pose_relative_tcp": [dx, dy, dz, dRx, dRy, dRz]
}
```

**Variedad requerida** (rangos del `sample_config.yaml`):

| Variable                  | Rango                               |
| ------------------------- | ----------------------------------- |
| `task_board.yaw`        | −30° a +30°                      |
| `task_board.x/y`        | ±0.1m respecto a posición nominal |
| `nic_rail` translation  | [−0.0215, +0.0234] m               |
| `sc_rail` translation   | [−0.06, +0.055] m                  |
| Iluminación (dome light) | 1500–3500 lux, color aleatorio     |
| Tipo de conector          | SFP y SC por separado               |

**Volumen objetivo:** ~5000 escenas × 3 cámaras = **15k imágenes**

---

### Dataset B — IL masivo con CheatCode (sintético, automático)

**Para qué:** Entrenar la Capa 2 (ACT). Dataset principal de behavioral cloning.
El CheatCode genera episodios perfectos gratis con `ground_truth:=true`.

**Estructura — formato LeRobot dataset:**

| Campo                                 | Dims        | Descripción                                         |
| ------------------------------------- | ----------- | ---------------------------------------------------- |
| `observation.images.left_camera`    | 3×256×288 | RGB normalizado ImageNet                             |
| `observation.images.center_camera`  | 3×256×288 | Ídem                                                |
| `observation.images.right_camera`   | 3×256×288 | Ídem                                                |
| `observation.state.tcp_pose`        | 7           | xyz + quaternion xyzw                                |
| `observation.state.tcp_velocity`    | 6           | linear xyz + angular xyz                             |
| `observation.state.joint_positions` | 7           | 6 joints arm + gripper                               |
| `observation.state.ft_tared`        | 6           | fx,fy,fz,tx,ty,tz post-tare                          |
| `observation.state.keypoint_pose`   | 6           | Output de Capa 0 en ese frame                        |
| `action`                            | 6           | Twist vx,vy,vz,ωx,ωy,ωz del CheatCode             |
| `meta.task_prompt`                  | str         | Ej: "Insert sfp into sfp_port_0 on nic_card_mount_2" |
| `meta.insertion_event`              | bool        | Si el episodio terminó en inserción exitosa        |

**Estado total:** 7+6+7+6+6 = **32D**. Normalización MEAN_STD sobre el dataset completo.

**Volumen objetivo:** 600 eps SFP + 400 eps SC = **~1000 episodios**

---

### Dataset C — Demostraciones humanas (teleoperation)

**Para qué:** Complementar Dataset B con micro-correcciones y recovery que el
CheatCode nunca genera. El CheatCode es perfecto pero no titubea — las demos
humanas cubren la distribución de errores reales.

**Fuente:** `lerobot-record` con teclado. Mismo formato que Dataset B.

**Volumen objetivo:** 300–500 episodios

**Ratio final para entrenar ACT:** 70% Dataset B + 30% Dataset C

---

### Dataset D — Buffer de inicialización RLPD (subconjunto de B+C)

**Para qué:** Inicializar el replay buffer del SAC residual (Capa 3) con
transiciones en la zona de inserción fina (<5mm). No requiere grabación adicional
— se filtra desde B+C automáticamente.

**Estructura de transición:**

```
(state_t[32D], delta_action_t[3D], reward_t, state_{t+1}[32D], done)
```

**Volumen estimado:** ~80k transiciones (últimos 50 frames de episodios
donde `dist_plug_tip ≤ 5mm`)

> **Límite de evaluación:** `/scoring/insertion_event` está **bloqueado**
> por el Zenoh ACL para el contenedor `model` (confirmado en
> `docker/aic_eval/aic_zenoh_config.json5`). Este topic se usa solo durante
> entrenamiento local con `AIC_ENABLE_ACL` desactivado. El SAC entrenado
> no depende de él en inferencia.

---

## 3. Comandos para crear los datasets

### Prerequisitos

```bash
# Desde WSL
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic
xhost +local:docker
export DBX_CONTAINER_MANAGER=docker
```

---

### 3.1 Dataset A — Ground truth keypoints

La recolección usa **dos contenedores Docker** orquestados con
`docker/docker-compose.dataset_a.yaml`:

- **`eval`** — imagen oficial `aic_eval`, levanta Gazebo con
  `ground_truth:=true` y sin engine. Publica `/tf` con la pose exacta de cada
  puerto SFP y SC.
- **`collector`** — imagen basada en `aic_eval` + `transforms3d`. Corre
  `scripts/collect_keypoint_dataset.py` dentro del mismo namespace de red que
  `eval` (`network_mode: service:eval`), por lo que tanto el router Zenoh como
  Gazebo Transport están accesibles en `localhost`.

#### Prerequisitos (una sola vez por sesión WSL)

WSL2 no activa `shared` mount propagation por defecto. Sin esto, Docker Desktop
no puede conectar los namespaces de red entre contenedores:

```bash
# Desde el host WSL (Giskard), NO dentro de ningún contenedor
sudo mount --make-rshared / && sudo mount --make-rshared /dev/pts
xhost +local:docker
```

Para que se aplique automáticamente en cada arranque de WSL:

```bash
sudo tee /etc/wsl.conf > /dev/null <<'EOF'
[boot]
command = mount --make-rshared / && mount --make-rshared /dev/pts
EOF
# Luego desde PowerShell: wsl --shutdown  (una sola vez)
```

#### Build del collector (solo cuando cambie el Dockerfile)

```bash
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic
docker compose -f docker/docker-compose.dataset_a.yaml build collector
```

El Dockerfile instala `transforms3d` sobre `aic_eval:latest` y copia
`docker/aic_collector/collect_entrypoint.sh`, que:

1. Hace `source /ws_aic/install/setup.bash`
2. Configura `ZENOH_CONFIG_OVERRIDE` para conectar al router del contenedor `eval`
3. Espera `${COLLECTOR_INIT_DELAY:-20}` segundos a que Gazebo inicialice
4. Ejecuta `python3 /workspace/collect_keypoint_dataset.py "$@"`

#### Ejecución — colección completa

```bash
# 5000 escenas × 2 tipos (sfp + sc) = ~10 000 muestras
docker compose -f docker/docker-compose.dataset_a.yaml up
```

Señales esperadas en los logs:

```
collector  | [collector] Waiting 20s for eval container to initialize...
collector  | [collector] Starting dataset collection. Args: ...
collector  | Waiting for camera_infos (timeout 60 s)...
collector  | [  50 samples |  0.5%] scene=24 type=sfp port=task_board/nic_card_mount_0/sfp_port_0_link
```

El contenedor `collector` termina solo cuando alcanza `--n_scenes`. El
contenedor `eval` puede detenerse manualmente con `Ctrl+C`.

#### Test rápido con pocas escenas

```bash
N_SCENES=100 docker compose -f docker/docker-compose.dataset_a.yaml up
```

---

#### Dónde se guardan los resultados

El volumen bind-mounted en el compose es:

```
${HOME}/aic_datasets/dataset_A   ←→   /aic_datasets/dataset_A  (dentro del collector)
```

En WSL esto corresponde a `~/aic_datasets/dataset_A` (por defecto
`/home/diego/aic_datasets/dataset_A`). Para usar otra ruta:

```bash
DATASET_DIR=/ruta/personalizada docker compose -f docker/docker-compose.dataset_a.yaml up
```

Estructura de salida por muestra:

```
~/aic_datasets/dataset_A/
  ├── 0000000.json          ← metadata: keypoints 2D × 3 cámaras + pose_relative_tcp [6D]
  ├── 0000001.json
  ├── ...
  └── images/
        ├── 0000000_left.png
        ├── 0000000_center.png
        ├── 0000000_right.png
        ├── 0000001_left.png
        └── ...
```

Cada JSON contiene:

```json
{
  "sample_id": 0,
  "scene_idx": 0,
  "connector_type": "sfp",
  "port_frame": "task_board/nic_card_mount_0/sfp_port_0_link",
  "images": {
    "left":   "images/0000000_left.png",
    "center": "images/0000000_center.png",
    "right":  "images/0000000_right.png"
  },
  "keypoints_left":   [[u1,v1], "...", [u9,v9]],
  "keypoints_center": [[u1,v1], "...", [u9,v9]],
  "keypoints_right":  [[u1,v1], "...", [u9,v9]],
  "pose_relative_tcp": [dx, dy, dz, dRx, dRy, dRz],
  "tcp_pose_in_base":  [x, y, z, qx, qy, qz, qw],
  "port_pose_in_base": [x, y, z, qx, qy, qz, qw]
}
```

> **Permisos:** Los archivos se crean como `root` dentro del contenedor.
> Si aparecen errores de permiso en el host, ejecutar:
> `sudo chown -R diego:diego ~/aic_datasets`

---

#### Cómo ver y verificar el dataset generado

```bash
# Contar muestras totales
ls ~/aic_datasets/dataset_A/*.json | wc -l
# Objetivo: ~10 000 (5000 escenas × 2 tipos)

# Estadísticas de pose relativa (primeras 100 muestras)
python3 - <<'EOF'
import json, glob, numpy as np
files = sorted(glob.glob('/home/diego/aic_datasets/dataset_A/*.json'))[:100]
poses = [json.load(open(f))['pose_relative_tcp'] for f in files]
arr = np.array(poses)
print(f'Muestras:          {len(arr)}')
print(f'pose_rel mean:     {arr.mean(0).round(4)}')
print(f'pose_rel std:      {arr.std(0).round(4)}')
print(f'dist XYZ medio:    {np.linalg.norm(arr[:,:3], axis=1).mean():.4f} m')
EOF

# Verificar integridad de imágenes de la primera muestra
python3 - <<'EOF'
import json, pathlib
BASE = pathlib.Path('/home/diego/aic_datasets/dataset_A')
s = json.load(open(BASE / '0000000.json'))
for cam, rel in s['images'].items():
    p = BASE / rel
    size_kb = p.stat().st_size // 1024 if p.exists() else 0
    print(f'{cam:8s}  {size_kb:5d} KB  {"✓" if p.exists() else "✗ MISSING"}')
print('connector_type:    ', s['connector_type'])
print('port_frame:        ', s['port_frame'])
print('keypoints_center[0]:', s['keypoints_center'][0])
print('pose_relative_tcp: ', [round(v,4) for v in s['pose_relative_tcp']])
EOF

# Balance entre tipos de conector
python3 - <<'EOF'
import json, glob
from collections import Counter
files = glob.glob('/home/diego/aic_datasets/dataset_A/*.json')
c = Counter(json.load(open(f))['connector_type'] for f in files)
for k,v in sorted(c.items()):
    print(f'{k:6s}: {v:6d} muestras')
EOF

# Ver una imagen con el visor del sistema
xdg-open ~/aic_datasets/dataset_A/images/0000000_center.png
```

---

### 3.2 Dataset B — Calibración F/T para el estimador Bayesiano

**Para qué:** Generar los datos que permiten calibrar los parámetros
(`GAIN_X`, `GAIN_Y`, `GAIN_Rz`, `BIAS_*`) del estimador Bayesiano de contacto.
La policy `FTCalibrationSampler` posiciona el cable tip en 9 offsets XY
controlados respecto al puerto, manteniendo cada posición 30 s para que el
sensor F/T se estabilice.

**Duración:** 9 posiciones × 30 s = ~4.5 min de tiempo simulado.
Con la física del cable activa el simulador puede correr hasta ~77× más lento
que el tiempo real; estimar ~6-7 h de tiempo real para la secuencia completa.
Parar antes no invalida el bag — cada fase completa es un punto de datos útil.

> **Prerequisito para los terminales 2, 3 y 4**
>
> `pixi` no está disponible en WSL. Ejecutar esto **al inicio de cada terminal**:
>
> ```bash
> distrobox enter -r aic_eval
> # Primera vez: pide crear contraseña sudo dentro del contenedor
>
> # Dentro del contenedor (prompt cambia a diego@aic_eval):
> source /ws_aic/install/setup.bash
> export RMW_IMPLEMENTATION=rmw_zenoh_cpp
> export ZENOH_CONFIG_OVERRIDE=';transport/shared_memory/enabled=false'
> ```

**Prerequisito — copiar la policy al workspace del contenedor (una sola vez):**

```bash
# Dentro del contenedor (Terminal 2, 3 o 4 con el entorno ya configurado)
sudo cp \
  /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic/aic_example_policies/aic_example_policies/ros/FTCalibrationSampler.py \
  /ws_aic/install/lib/python3.12/site-packages/aic_example_policies/ros/
```

> Con Docker (`docker-compose.dataset_b.yaml`) este paso NO es necesario —
> el servicio `tools` monta el source directamente.

---

**Terminal 1 — Simulación con ground truth y engine activo:**

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
  gazebo_gui:=false \
  ground_truth:=true \
  start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true \
  model_discovery_timeout_seconds:=60
```

**Terminal 2 — Tare del sensor F/T (antes de arrancar):**

```bash
distrobox enter -r aic_eval
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE=';transport/shared_memory/enabled=false'

ros2 service call \
  /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
# Respuesta esperada: success=True, message='Successfully tared force torque sensor.'
```

**Terminal 3 — Grabar bag de calibración:**

> Los bags se crean en `/tmp/` para evitar problemas de permisos del contenedor
> rootful. Moverlos después con `sudo chown -R diego:diego ~/aic_datasets`.

```bash
distrobox enter -r aic_eval
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE=';transport/shared_memory/enabled=false'

ros2 bag record \
  --topics \
    /fts_broadcaster/wrench \
    /joint_states \
    /tf \
    /tf_static \
  -o /tmp/ft_calib_$(date +%Y%m%d_%H%M%S)
```

**Terminal 4 — Ejecutar FTCalibrationSampler:**

```bash
distrobox enter -r aic_eval
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE=';transport/shared_memory/enabled=false'

ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.FTCalibrationSampler
```

Señales esperadas en los logs:

```
[aic_model]: Goal accepted
[aic_model]: FTCalibrationSampler.insert_cable() task=...cable_name='cable_0'...
[aic_model]: --- OFFSET [1/9]: free_space dx=0.0mm  dy=0.0mm  z=0.120m ---
[aic_model]: Manteniendo 'free_space' durante 30 s...
[aic_model]: 'free_space' completado.
[aic_model]: --- OFFSET [2/9]: y_plus_15mm dx=0.0mm  dy=15.0mm  z=0.002m ---
...
```

El engine imprime `insert_cable execute loop` cada ~77 s (heartbeat).
Parar con **Ctrl+C** en ambas terminales (3 y 4) cuando se hayan completado
las fases deseadas o al final de la secuencia.

---

#### Alternativa — Docker puro (sin distrobox)

Usar `docker/docker-compose.dataset_b.yaml`. **No requiere copiar la policy**
manualmente — el servicio `tools` monta el source del repo directamente.

```bash
# Preparar directorio de salida
mkdir -p ~/aic_bags

# Levantar simulación + contenedor de herramientas
BAGS_DIR=~/aic_bags \
  docker compose -f docker/docker-compose.dataset_b.yaml up -d eval tools

# Cada terminal (2, 3, 4) abre una shell en tools:
docker exec -it aic_dataset_b-tools-1 bash
```

Desde esa shell los comandos son idénticos a los del flujo distrobox
(sin el bloque prerequisito — ya configurado en el entrypoint):

```bash
# Terminal 2 — Tare
ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger

# Terminal 3 — Bag (va a /bags = ~/aic_bags en el host)
ros2 bag record \
  --topics \
    /fts_broadcaster/wrench \
    /joint_states \
    /tf \
    /tf_static \
  -o /bags/ft_calib_$(date +%Y%m%d_%H%M%S)

# Terminal 4 — FTCalibrationSampler
ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.FTCalibrationSampler
```

```bash
# Detener todo al finalizar
docker compose -f docker/docker-compose.dataset_b.yaml down
```

---

**Post-procesamiento — extraer bag y calibrar:**

```bash
# 1. Extraer el bag MCAP a CSV (dentro del contenedor, usar python3)
python3 tools/extract_eval_bags_to_csv.py /tmp/ft_calib_TIMESTAMP

# 2. Calcular GAIN y BIAS (desde Windows, sin contenedor)
python tools/calibrate_ft.py \
  --calib extracted_eval_bags_csv/ft_calib_TIMESTAMP

# Combina datos con bag_trial_* anteriores y exporta tools/calibration_results.csv
```

---

### 3.3 Dataset C — Teleoperation humana

**Terminal 1 — Simulación sin ground truth (condición de evaluación):**

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
  gazebo_gui:=true \
  launch_rviz:=true \
  ground_truth:=false \
  start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true \
  model_discovery_timeout_seconds:=600
```

**Terminal 2 — Tare antes de cada episodio:**

```bash
pixi run ros2 service call \
  /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```

**Terminal 3 — Grabar con lerobot-record (SFP):**

```bash
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian \
  --robot.teleop_frame_id=gripper/tcp \
  --dataset.repo_id=local/aic_teleop_sfp \
  --dataset.single_task="Insert SFP module into SFP port on NIC card" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

**Terminal 3 — Grabar con lerobot-record (SC):**

```bash
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian \
  --robot.teleop_frame_id=gripper/tcp \
  --dataset.repo_id=local/aic_teleop_sc \
  --dataset.single_task="Insert SC plug into SC port on task board" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

**Controles de teclado:**

| Tecla             | Movimiento                    |
| ----------------- | ----------------------------- |
| `w/s`           | −/+ linear Y                 |
| `a/d`           | −/+ linear X                 |
| `r/f`           | −/+ linear Z                 |
| `Shift+W/S/A/D` | Rotación X/Y                 |
| `q/e`           | −/+ rotación Z              |
| `t`             | Alternar lento/rápido        |
| `→`            | Confirmar episodio, siguiente |
| `←`            | Cancelar y re-grabar          |
| `ESC`           | Parar grabación              |

**Verificación:**

```bash
pixi run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/aic_teleop_sfp')
print(f'Episodios: {ds.num_episodes}')
print(f'Frames: {len(ds)}')
print(f'FPS: {ds.fps}')
print(f'Features: {list(ds.features.keys())}')
"
```

---

### 3.4 Dataset D — Filtrado para RLPD

```bash
pixi run python scripts/filter_insertion_frames.py \
  --input_dataset local/aic_cheatcode_dataset \
  --output_path ~/aic_datasets/dataset_D.hdf5 \
  --dist_threshold 0.005 \
  --last_n_frames 50

# Verificar
pixi run python -c "
import h5py
with h5py.File('$HOME/aic_datasets/dataset_D.hdf5', 'r') as f:
    print(f'Transiciones: {len(f[\"state\"])}')
    print(f'Keys: {list(f.keys())}')
"
```

---

## 4. Entrenamiento

### Capa 0 — Keypoint Estimator

```bash
pixi run python scripts/train_keypoint_estimator.py \
  --dataset_dir ~/aic_datasets/dataset_A \
  --connector_type sfp \
  --backbone mobilenet_v3_small \
  --epochs 50 --batch_size 32 \
  --output_dir outputs/keypoint_sfp

pixi run python scripts/train_keypoint_estimator.py \
  --dataset_dir ~/aic_datasets/dataset_A \
  --connector_type sc \
  --backbone mobilenet_v3_small \
  --epochs 50 --batch_size 32 \
  --output_dir outputs/keypoint_sc
```

### Capa 2 — ACT con estado extendido [32D]

```bash
pixi run lerobot-train \
  --dataset.repo_id=local/aic_cheatcode_dataset \
  --policy.type=act \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --output_dir=outputs/train/act_aic_v1 \
  --job_name=act_aic_v1 \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=local/act_aic_policy
```

### Verificación de scoring local

```bash
# Terminal 1
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=false start_aic_engine:=true

# Terminal 2
AIC_RESULTS_DIR=~/aic_results/mi_politica_v1 \
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=mi_politica.MyPolicy

# Ver resultados
cat ~/aic_results/mi_politica_v1/scoring.yaml
```

---

## 5. Verificación final con docker-compose

```bash
xhost +local:docker

# Build de la imagen del modelo
docker compose -f docker/docker-compose.yaml build model

# Levantar eval + model (replica portal de evaluación)
docker compose -f docker/docker-compose.yaml up

# Ver scores de los 3 trials
docker compose -f docker/docker-compose.yaml logs eval \
  | grep -E "score|Score|tier|Tier|trial|Trial"
```

> Antes de subir al portal, cambiar en `docker-compose.yaml`:
> `ground_truth:=true` → `ground_truth:=false`

---

## 6. Resumen — datasets por capa y volumen

| Dataset                     | Para qué                  | Fuente                        | Volumen           | Esfuerzo humano |
| --------------------------- | -------------------------- | ----------------------------- | ----------------- | --------------- |
| **A — Keypoints**    | Capa 0 pose estimator      | Gazebo GT `/tf` automático | ~15k imágenes    | Ninguno         |
| **B — IL CheatCode** | Capa 2 ACT (70%)           | CheatCode automático         | ~1000 episodios   | Ninguno         |
| **C — Teleop**       | Capa 2 ACT (30%)           | Teleoperation humana          | ~400 episodios    | Moderado        |
| **D — RLPD buffer**  | Capa 3 SAC inicialización | Filtrado de B+C               | ~80k transiciones | Ninguno         |

---

## 7. Advertencias y límites

1. **`/scoring/**` bloqueado en evaluación** (`aic_zenoh_config.json5`): Usar
   durante entrenamiento local únicamente. El SAC no lo necesita en inferencia.
2. **Tare del F/T solo durante entrenamiento**: La doc confirma que el scoring
   llama al tare automáticamente antes de spawnear el cable. Consistente con
   el field `ft_tared[6]` del vector de estado.
3. **`ground_truth:=true` en `docker-compose.yaml`** está activo en la config
   actual — cambiar a `false` antes de enviar al portal.
4. **Domain gap sim→real para Fase 2**: El Keypoint Estimator entrenado en
   Gazebo necesitará fine-tuning o domain randomization para el workcell físico.
   Los scripts de `events.py` en Isaac Lab (`randomize_dome_light`) proveen la
   base para esto.
5. **Isaac Lab** disponible en `aic_utils/aic_isaac/` con `randomize_board_and_parts`
   ya implementado — alternativa para generación masiva de Dataset A con mayor
   variedad de iluminación que Gazebo.
