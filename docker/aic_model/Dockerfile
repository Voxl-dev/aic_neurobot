###############################################################################
# AIC Model – imagen de producción
#
# ANTES DE HACER docker build, deben existir:
#   models/act_policy/config.json          ← generado por train_act_fast.py
#   models/act_policy/model.safetensors    ← generado por train_act_fast.py
#   models/keypoints/best_sc.pt            ← checkpoint keypoints SC
#   models/keypoints/best_sfp.pt           ← checkpoint keypoints SFP
#   bayesiano/calibration_results.csv      ← generado por calibrate_ft.py
#                                             (opcional; usa defaults si falta)
#
# Build (desde la raíz del repo):
#   docker build -f docker/aic_model/Dockerfile -t aic_model:latest .
#
# Push a DockerHub:
#   docker tag aic_model:latest <tu_usuario>/aic_model:latest
#   docker push <tu_usuario>/aic_model:latest
###############################################################################

FROM docker.io/library/ros:kilted-ros-core

# ── Herramientas del sistema ──────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ── Pixi ─────────────────────────────────────────────────────────────────────
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# ── Código fuente ROS ─────────────────────────────────────────────────────────
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model            /ws_aic/src/aic/aic_model
COPY aic_interfaces       /ws_aic/src/aic/aic_interfaces
COPY aic_utils            /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock  /ws_aic/src/aic/
COPY pixi_env_setup.sh    /ws_aic/src/aic/

# ── Dependencias Python + ROS via pixi ───────────────────────────────────────
# El loop reintenta hasta 8 veces para tolerar hash-mismatch en builds locales.
RUN --mount=type=cache,target=/ws_aic/src/aic/.pixi/bld,sharing=locked \
    cd /ws_aic/src/aic && \
    for i in 1 2 3 4 5 6 7 8; do \
        echo "=== pixi install pass $i ===" ; \
        pixi install --locked && break || true ; \
    done && \
    pixi install --locked

# ── Artefactos de modelos (baked-in, sin parámetros de entorno necesarios) ───
#
#  ACT policy checkpoint (config.json + model.safetensors)
COPY models/act_policy   /opt/aic/models/act_policy

#  Keypoint estimators (.pt)
COPY models/keypoints    /opt/aic/models/keypoints

#  Calibración Bayesiana F/T (opcional – fallback a defaults si está vacío)
COPY bayesiano           /opt/aic/bayesiano

# ── Variables de entorno baked-in ─────────────────────────────────────────────
# Rutas de modelos – absolutas, sin necesidad de ningún parámetro externo
ENV AIC_POLICY_CHECKPOINT=/opt/aic/models/act_policy
ENV AIC_KP_SC=/opt/aic/models/keypoints/best_sc.pt
ENV AIC_KP_SFP=/opt/aic/models/keypoints/best_sfp.pt
ENV AIC_BAYESIAN_DIR=/opt/aic/bayesiano

# Resolución de imágenes: debe coincidir con la resolución usada en entrenamiento.
# Si se entrenó con train_act_fast.py (--fast): 240x320
# Si se entrenó full (lerobot-train, 480x640): cambiar a 480 y 640
ENV AIC_IMG_H=480
ENV AIC_IMG_W=640

# ── Entrypoint ────────────────────────────────────────────────────────────────
WORKDIR /ws_aic/src/aic

COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# AIC_ROUTER_ADDR es inyectado por la infraestructura de evaluación (AWS/AIC Engine)
if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "ERROR: AIC_ROUTER_ADDR must be provided by the evaluation environment"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

# Credenciales Zenoh
if should_enable_acl; then
  if [[ -z "$AIC_MODEL_PASSWD" ]]; then
    echo "ERROR: AIC_MODEL_PASSWD must be provided when AIC_ENABLE_ACL=true"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# Construir la cadena de configuración Zenoh
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

RUN sed -i 's/\r$//' /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_model.NeuroPolicy", "-p", "use_sim_time:=true"]
