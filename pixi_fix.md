# Fix: pixi install hash mismatch en Docker

## Problema principal

Al construir la imagen Docker del model (`my-solution:v1`), `pixi install --locked`
fallaba con el siguiente error:

```
hash mismatch when extracting file:///ws_aic/src/aic/.pixi/bld/
  ros-kilted-aic-task-interfaces/L8Ick2hzOx4/output/linux-64/
  ros-kilted-aic-task-interfaces-0.0.1-hb0f4dca_0.conda
expected 57f909042fc82ba..., got 3a3c1cd0a941f0b5...
```

### Causa raíz

`pixi-build-ros` lanza una segunda sesión interna de `rattler-build` para construir
el entorno de compilación (`build-dependencies`) de paquetes como
`aic_example_policies`. Esta sesión interna recompila paquetes locales (como
`aic_task_interfaces`) en el **mismo directorio de salida determinístico**
(`L8Ick2hzOx4/`) que ya usó la sesión externa.

El directorio de salida es determinístico (hash del recipe), por lo que ambas
sesiones escriben en la misma ruta. La segunda escritura sobreescribe el
`.conda` con un hash diferente (entorno de compilación distinto → `hash_input.json`
diferente → hash del archivo diferente), pero el session cache de rattler en memoria
sigue esperando el hash original → **hash mismatch**.

El problema es **within-session** (dentro de una sola ejecución de `pixi install`),
no entre builds de Docker. Vaciar el cache de Docker no lo resuelve.

---

## Solución aplicada: loop de reintentos con cache persistente

**Archivo modificado:** `docker/aic_model/Dockerfile`

```dockerfile
SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/ws_aic/src/aic/.pixi/bld,sharing=locked \
    cd /ws_aic/src/aic && \
    for i in 1 2 3 4 5 6 7 8; do \
        echo "=== pixi install pass $i ===" ; \
        pixi install --locked && break || true ; \
    done && \
    pixi install --locked
```

### Por qué funciona

- La primera pasada falla en el primer paquete que sufre el doble-build (ej.
  `aic_task_interfaces`), pero deja el archivo `.conda` con el hash "estable" B
  en `.pixi/bld/`.
- El `--mount=type=cache` hace que `.pixi/bld/` persista entre todas las pasadas
  dentro del mismo `RUN`.
- La segunda pasada encuentra el `.conda` ya existente, `pixi-build-ros` lo detecta
  y **salta el rebuild**. El session cache registra el hash B para ese paquete.
  Luego falla en el siguiente paquete de la cadena.
- El proceso se repite: cada pasada estabiliza un paquete más en el cache.
- Con hasta 8 pasadas (hay 5 paquetes locales con build-deps), todos los paquetes
  quedan estabilizados y el `pixi install --locked` final tiene éxito.

---

## Fix adicional: CRLF en entrypoint.sh

**Problema:** El Dockerfile se guarda en Windows con saltos de línea `\r\n` (CRLF).
El heredoc inline copia esos `\r` al script `/entrypoint.sh` dentro del contenedor.
El kernel de Linux intenta ejecutar `#!/bin/bash\r`, que no existe →
`exec /entrypoint.sh: no such file or directory` aunque el archivo sí exista.

**Fix aplicado en `docker/aic_model/Dockerfile`:**

```dockerfile
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
...
EOF

RUN sed -i 's/\r$//' /entrypoint.sh   # elimina CRLF del shebang y demás líneas
```

---

## Cambios en docker/docker-compose.yaml

### Display X11 para Gazebo (interfaz gráfica desde WSL)

Para que Gazebo pueda mostrar su ventana gráfica al correr desde WSL, se agregaron
una variable de entorno y un volumen al servicio `eval`:

```yaml
services:
  eval:
    ...
    environment:
      AIC_EVAL_PASSWD: CHANGE_IN_PROD
      AIC_MODEL_PASSWD: CHANGE_IN_PROD
      DISPLAY: $DISPLAY            # pasa el display X11 del host al contenedor
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix   # socket X11 del host
```

---

## Comandos para correr desde WSL

Hay dos modos: **todo en Docker** (replica exacta del entorno de evaluación oficial)
o **eval en distrobox + política en pixi** (ciclo de desarrollo más rápido).

---

### Modo A — Todo en Docker (evaluación oficial local)

Replica exactamente el entorno del portal de evaluación, incluyendo Zenoh ACL.

```bash
# En WSL (no PowerShell)
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic

# Autorizar conexiones X11 desde Docker (una vez por sesión WSL)
xhost +local:docker

# (Opcional) Reconstruir la imagen del modelo si cambiaste código
docker compose -f docker/docker-compose.yaml build model

# Bajar contenedores anteriores si estaban corriendo
docker compose -f docker/docker-compose.yaml down

# Levantar ambos contenedores (eval + model) en foreground para ver logs
docker compose -f docker/docker-compose.yaml up

# O en background:
docker compose -f docker/docker-compose.yaml up -d
docker compose -f docker/docker-compose.yaml logs -f   # ver logs en tiempo real
```

**Ver logs de un servicio específico:**
```bash
docker compose -f docker/docker-compose.yaml logs -f eval
docker compose -f docker/docker-compose.yaml logs -f model
```

**Parar y limpiar:**
```bash
docker compose -f docker/docker-compose.yaml down
```

> **Nota:** `xhost +local:docker` debe ejecutarse en cada sesión de WSL nueva,
> antes de `docker compose up`.

---

### Modo B — Eval en distrobox + política en pixi (desarrollo iterativo)

Más rápido para iterar: evita reconstruir la imagen Docker en cada cambio de código.
El contenedor `aic_eval` corre la simulación y el scoring; la política corre directamente
con pixi en el host (fuera de Docker).

**Terminal 1 — Levantar la simulación (eval container):**
```bash
# En WSL
export DBX_CONTAINER_MANAGER=docker

# Primera vez: crear el contenedor de distrobox
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval

# Entrar y lanzar el entorno (con GUI de Gazebo y RViz)
distrobox enter -r aic_eval -- /entrypoint.sh \
  gazebo_gui:=true \
  launch_rviz:=true \
  ground_truth:=false \
  start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true \
  model_discovery_timeout_seconds:=600

# Sin GUI (más liviano, headless):
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=false \
  start_aic_engine:=true
```

**Terminal 2 — Correr la política con pixi (fuera del contenedor):**
```bash
# En WSL, desde la raíz del repo
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic

# Si cambiaste código Python de tu paquete, reinstalar antes de correr
pixi reinstall ros-kilted-mi-politica   # reemplazar con el nombre de tu paquete

# Correr la política de ejemplo (WaveArm)
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.WaveArm

# Correr la política ACT de referencia
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunACT

# Correr CheatCode (requiere ground_truth:=true en Terminal 1)
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode

# Correr tu política propia
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=mi_politica.MyPolicy
```

**Terminal 3 (opcional) — Teleoperation para recolectar datos:**
```bash
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic

# Tare del F/T sensor ANTES de cada episodio
pixi run ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger

# Teleoperar con teclado (espacio cartesiano, frame base)
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --display_data=true

# Grabar dataset de entrenamiento
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=tu_usuario_hf/aic_dataset \
  --dataset.single_task="Insert SFP cable into NIC port" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

**Entrenar el modelo (después de recolectar datos):**
```bash
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic

pixi run lerobot-train \
  --dataset.repo_id=tu_usuario_hf/aic_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_aic \
  --job_name=act_aic \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=tu_usuario_hf/act_aic_policy
```

---

### Tabla resumen: cuándo usar cada modo

| Situación | Modo recomendado |
|---|---|
| Verificar antes de enviar al portal | **Modo A** (Docker completo) |
| Iterar rápido sobre código de política | **Modo B** (distrobox + pixi) |
| Recolectar datos de teleoperation | **Modo B** Terminal 3 |
| Entrenar el modelo | **Modo B** (pixi directo) |
| Test con Zenoh ACL activo | **Modo A** (descomentar `AIC_ENABLE_ACL: true` en docker-compose.yaml) |

---

## Resumen de archivos modificados

| Archivo | Cambio |
|---|---|
| `docker/aic_model/Dockerfile` | Loop de reintentos para `pixi install` + `sed` CRLF fix |
| `docker/docker-compose.yaml` | `DISPLAY` env var + volumen `/tmp/.X11-unix` para Gazebo en WSL |
