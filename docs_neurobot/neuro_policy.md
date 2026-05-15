# NeuroPolicy — Guía de referencia

Política de producción para el AIC Challenge. Usa ACT (Action Chunking Transformer) para todo el movimiento de inserción de cable.

---

## 1. Archivos del policy

| Archivo | Descripción |
|---|---|
| `aic_model/aic_model/NeuroPolicy.py` | Política principal — ACT + FSM + Bayesian |
| `aic_model/aic_model/keypoint_step4.py` | Estimador de keypoints (MobileNetV3-Small, 9 puntos) |
| `aic_model/aic_model/bayesian_estimator.py` | Estimador bayesiano F/T (Axia80) |
| `aic_model/aic_model/insertion_state_machine.py` | FSM: APPROACH → CONTACT → EXPLORE → SUCCESS/RETRACT |
| `aic_model/aic_model/policy.py` | Clase base `Policy` |
| `docker/aic_model/Dockerfile` | Imagen de producción (modelos baked-in) |

---

## 2. Cómo funciona

### Loop principal (10 Hz)

```
on_configure()
  └─ _load_act(AIC_POLICY_CHECKPOINT)        ← ACT modelo
  └─ KeypointEstimatorBank(sc, sfp)           ← keypoints
  └─ Axia80BayesianEstimator(calibration.csv) ← F/T bayesiano

insert_cable(task, get_observation, move_robot, send_feedback)
  ├─ conn = task.port_type  # "sc" o "sfp"
  ├─ time_limit = task.time_limit  # segundos, respetado estrictamente
  └─ loop (hasta fsm.done o time_limit):
       obs = get_observation()
       ft  = obs.wrist_wrench  → Axia80BayesianEstimator.update(ft) → confident
       state = fsm.step(ft, t, bayesian_confident=confident)
       if state == RETRACT:
           act.reset(); send zero velocity; sleep 0.5s
       else:
           batch = _act_batch(obs, conn)  # 3 imágenes + state 32D
           action = act.select_action(batch)  # 6D twist
           move_robot(velocity=action, mode=MODE_VELOCITY)
```

### Componentes

| Componente | Rol |
|---|---|
| **ACT** | Genera el twist Cartesiano `[vx,vy,vz,wx,wy,wz]` en `base_link` |
| **KeypointEstimatorBank** | Predice 9 keypoints 2D en la cámara central, normalizados a [0,1] |
| **InsertionFSM** | Detecta CONTACT (Fz > 1N), SUCCESS (click: dFz + Tz spike), RETRACT (Fz > 24N por 0.5s) |
| **BayesianEstimator** | Filtra vibraciones F/T, decide si el alineamiento es confiable antes de avanzar |

### Vector de estado (32D)

El orden debe coincidir exactamente con `bag_to_lerobot.py`:

| Índices | Contenido | Dimensión |
|---|---|---|
| 0–5 | Joints del brazo UR5e | 6 |
| 6 | Gripper | 1 |
| 7–24 | Keypoints normalizados (9 puntos × 2 coords) | 18 |
| 25–27 | TCP posición (x, y, z) en `base_link` | 3 |
| 28–31 | TCP orientación (quaternion x,y,z,w) | 4 |

### Acción (6D)

Twist Cartesiano en `base_link`, modo `MODE_VELOCITY`:

```
[vx, vy, vz, wx, wy, wz]
```

---

## 3. Datos necesarios antes del build

### Estructura de directorios (desde la raíz del repo)

```
models/
  act_policy/
    config.json          ← generado por train_act_fast.py
    model.safetensors    ← generado por train_act_fast.py
  keypoints/
    best_sc.pt           ← checkpoint SC keypoints
    best_sfp.pt          ← checkpoint SFP keypoints
bayesiano/
  calibration_results.csv  ← generado por calibrate_ft.py (opcional)
```

> Si `calibration_results.csv` no existe, el policy usa los valores por defecto hardcodeados en `bayesian_estimator.py`.

---

## 4. Cómo cargar los pesos del modelo ACT

### Paso 1 — Preparar el dataset

```bash
cd fine_tune_act_v2/fine_tune_act
pixi run python scripts/bag_to_lerobot.py \
  --bags /ruta/a/bags/ \
  --output data/dataset_lerobot \
  --repo_id aic_team/sfp_sc_insertion \
  --ckpt_sc /ruta/keypoints/best_sc.pt \
  --ckpt_sfp /ruta/keypoints/best_sfp.pt
```

### Paso 2 — Entrenar ACT (modo fast, 240×320)

```bash
bash scripts/run_finetune.sh --fast
```

El checkpoint queda en `outputs/act_fast_run1/checkpoints/step_010000/`.

> Si se entrena con resolución completa (480×640), cambiar `AIC_IMG_H=480` y `AIC_IMG_W=640` en el Dockerfile.

### Paso 3 — Copiar pesos al repo

```bash
# ACT policy
cp -r outputs/act_fast_run1/checkpoints/step_010000/* ../../models/act_policy/

# Keypoints (ajustar rutas según el run)
cp fine_tune_act_v2/keypoints/keypoint_sc_*/best_sc.pt   models/keypoints/best_sc.pt
cp fine_tune_act_v2/keypoints/keypoint_sfp_v2/best_sfp.pt models/keypoints/best_sfp.pt
```

### Paso 4 — Calibración F/T (opcional)

```bash
pixi run python tools/calibrate_ft.py
# Genera bayesiano/calibration_results.csv
```

---

## 5. Cómo construir la imagen Docker

El Dockerfile bake los modelos dentro de la imagen — no se necesitan parámetros de entorno adicionales en AWS.

```bash
# Desde la raíz del repo
docker build -f docker/aic_model/Dockerfile -t aic_model:latest .
```

### Variables de entorno baked-in

| Variable | Valor baked-in | Descripción |
|---|---|---|
| `AIC_POLICY_CHECKPOINT` | `/opt/aic/models/act_policy` | Directorio del checkpoint ACT |
| `AIC_KP_SC` | `/opt/aic/models/keypoints/best_sc.pt` | Checkpoint SC |
| `AIC_KP_SFP` | `/opt/aic/models/keypoints/best_sfp.pt` | Checkpoint SFP |
| `AIC_BAYESIAN_DIR` | `/opt/aic/bayesiano` | Directorio calibración F/T |
| `AIC_IMG_H` | `240` | Alto de imagen para ACT |
| `AIC_IMG_W` | `320` | Ancho de imagen para ACT |

### Variables inyectadas en runtime por AWS (NO baked-in)

| Variable | Quién la inyecta |
|---|---|
| `AIC_ROUTER_ADDR` | Infraestructura de evaluación AWS |
| `AIC_MODEL_PASSWD` | Infraestructura de evaluación AWS |
| `ZENOH_ROUTER_CHECK_ATTEMPTS` | `docker-compose.yaml` local |

### Verificación local

```bash
docker compose -f docker/docker-compose.yaml up
```

---

## 6. Envío a AWS ECR

El registro de entrega es AWS ECR, **no DockerHub**. Las credenciales llegan por email de onboarding al líder del equipo.

### Requisito: AWS CLI instalado

```bash
# Verificar
aws --version
```

### Autenticación

```bash
# Configurar perfil (una sola vez)
aws configure --profile <team_name>
# Access Key ID:       (del email)
# Secret Access Key:   (del email)
# Default region:      us-east-1
# Output format:       json

# Activar el perfil en la sesión actual
export AWS_PROFILE=<team_name>

# Login a ECR (válido 12 horas)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    973918476471.dkr.ecr.us-east-1.amazonaws.com
```

### Tag y push

```bash
# Incrementar versión en cada envío — los tags en ECR son INMUTABLES
docker tag aic_model:latest \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1

docker push \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
```

### Registro en el portal

1. Copiar el URI completo: `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1`
2. Ir al portal de submissions → `AI for Industry Challenge` → `Submit`
3. Seleccionar fase `Qualification`
4. Pegar el URI en el campo `OCI Image`
5. Click `Submit`

> Hacer push a ECR sin registrar en el portal **no activa la evaluación**.

### Límites

- 1 submission por día
- Los tags son inmutables: si ya existe `:v1`, usar `:v2`, `:v3`, etc.
- Si el login de ECR expira (12 h), repetir el paso de autenticación

---

## 7. Flujo completo de submit (resumen)

```bash
# 1. Entrenar y copiar pesos (ver sección 4)

# 2. Build con modelos reales
docker build -f docker/aic_model/Dockerfile -t aic_model:latest .

# 3. Verificar localmente
docker compose -f docker/docker-compose.yaml up

# 4. Auth ECR
export AWS_PROFILE=<team_name>
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    973918476471.dkr.ecr.us-east-1.amazonaws.com

# 5. Tag (incrementar versión)
docker tag aic_model:latest \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1

# 6. Push
docker push \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1

# 7. Registrar en portal con el URI completo
```
