# Keypoint Estimator — Capa 0 para AIC

Esta etapa implementa el modelo propuesto del diagrama: un estimador visual de 9 keypoints 2D del puerto objetivo. La idea es que ACT no aprenda la alineacion completa solo desde RGB, sino que reciba una representacion geometrica mas estable.

## Objetivo

Entrada:
- Imagen RGB de una camara: `left`, `center` o `right`.
- Tipo de conector: `sfp` o `sc`.

Salida:
- 18 valores normalizados: `9 x (u, v)`.
- En pixeles, esos puntos representan los 9 keypoints del puerto objetivo.

## Decision MVP

- Dos modelos independientes: uno para `sfp` y otro para `sc`.
- Las tres camaras comparten el mismo modelo por tipo de conector.
- Backbone: `MobileNetV3-Small` preentrenado en ImageNet.
- Head: regresion directa a `18` valores.
- Loss: `SmoothL1Loss`, tambien conocida como Huber loss.

`SmoothL1Loss` es una perdida robusta: se comporta como L2 cerca del objetivo, pero reduce el castigo de errores grandes. Esto importa porque algunos keypoints pueden quedar cerca del borde, con ruido visual o con labels automaticos imperfectos.

## Dataset usado

Por defecto usamos el dataset unificado:

```bash
dataset_A/dataset_A_complete_merged/dataset_A_complete_merged
```

Cada JSON aporta hasta tres samples de entrenamiento: una imagen por camara y sus keypoints correspondientes.

El split se hace por `scene_idx`, no por imagen. Esto evita fuga de datos: imagenes casi iguales de la misma escena no deben caer simultaneamente en train y validacion.

## Entrenar los dos modelos

Desde la raiz del repo:

```bash
bash fine_tune_act_v2/fine_tune_act/scripts/run_keypoint_training.sh
```

Variables utiles:

```bash
DATASET_DIR=dataset_A/dataset_A_complete_merged/dataset_A_complete_merged \
OUTPUT_ROOT=outputs/keypoint_estimator \
EPOCHS=30 \
BATCH_SIZE=32 \
bash fine_tune_act_v2/fine_tune_act/scripts/run_keypoint_training.sh
```

Outputs esperados:

```text
outputs/keypoint_estimator/
├── sfp/
│   ├── best_sfp.pt
│   └── summary_sfp.json
└── sc/
    ├── best_sc.pt
    └── summary_sc.json
```

## Probar un checkpoint en una imagen

```bash
pixi run python fine_tune_act_v2/fine_tune_act/scripts/predict_keypoints.py \
  --checkpoint outputs/keypoint_estimator/sfp/best_sfp.pt \
  --image dataset_A/dataset_A_complete_merged/dataset_A_complete_merged/images/0000000_center.png \
  --output_json /tmp/keypoints_sfp.json \
  --save_debug /tmp/keypoints_sfp.png
```

## Paso 4: salida e integracion

El codigo del paso 4 queda encapsulado en:

```text
aic_example_policies/aic_example_policies/ros/keypoint_step4.py
```

Flujo implementado:

- Ejecuta el checkpoint de keypoints por camara (`left`, `center`, `right`).
- Usa la geometria 3D conocida del puerto `sfp` o `sc`.
- Resuelve PnP por camara para estimar `port_in_camera`.
- Transforma cada estimacion a `base_link` usando TF.
- Fusiona las tres estimaciones con pesos segun error de reproyeccion e inliers.
- Convierte el resultado a `pose_relative_tcp = [dx, dy, dz, dRx, dRy, dRz]`.
- En `RunACT.py`, concatena esos 6 valores al estado proprioceptivo de 26D para formar el estado 32D.

Variables de entorno utiles en runtime:

```bash
AIC_ENABLE_KEYPOINT_STEP4=1
AIC_KEYPOINT_SFP_CHECKPOINT=outputs/keypoint_estimator/sfp/best_sfp.pt
AIC_KEYPOINT_SC_CHECKPOINT=outputs/keypoint_estimator/sc/best_sc.pt
```

La integracion es defensiva: si el checkpoint no existe, si TF no entrega la
pose de camara, o si el checkpoint ACT cargado todavia espera un estado de 26D,
`RunACT` no se cae. En ese caso usa el estado legacy o rellena los 6 valores
visuales con ceros hasta que el entrenamiento/fine-tune 32D este listo.

## Criterio minimo antes de integrar con ACT

Para que esta capa sea util, no basta con que la loss baje. Debemos mirar:

- `val_mae_px`: error medio en pixeles.
- Preview visual sobre imagenes no vistas.
- Comparacion separada por `sfp` y `sc`.
- Casos donde los keypoints caen fuera del puerto o se invierte el orden.

Como regla practica inicial, si el error de validacion baja a pocos pixeles y las visualizaciones son coherentes, podemos usar estos modelos como fallback visual dentro de `bag_to_lerobot.py`.
