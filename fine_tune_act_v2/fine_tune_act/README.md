# Fine-tune ACT — Estructura lista para entrenar

**Para Juanse (RTX 2000 Ada 8GB) — handoff 7am**

Fine-tune de la policy ACT partiendo del checkpoint `lerobot/act_aloha_sim_insertion_scripted` (HuggingFace Hub), con **smart partial init** para evitar entrenar desde cero las capas con dimensiones incompatibles.

---

## ⚠️ ÚNICA cosa que debe integrarse antes de correr

**Dayana** debe pegar su lógica de extracción de keypoints en:
- Archivo: `scripts/bag_to_lerobot.py`
- Función: `extract_keypoints_from_tf()` (línea ~196, marcada con banner)
- Cambiar luego: `_KEYPOINT_FUNCTION_IS_PLACEHOLDER = False`

Sin esto, el script aborta con un error claro antes de generar un dataset inútil.

---

## Estructura

```
fine_tune_act/
├── README.md                      ← este archivo
├── KEYPOINT_ESTIMATOR.md          ← Capa 0: modelo visual de keypoints
├── pixi_deps.toml                 ← deps a pegar en pixi.toml del repo aic
├── configs/
│   └── act_aic.yaml               ← config Hydra del fine-tune
├── aic_model/
│   ├── __init__.py
│   └── load_act.py                ← descarga checkpoint HF + smart partial init
└── scripts/
    ├── bag_to_lerobot.py          ← bags ROS 2 → HF LeRobotDataset  ⚠ TODO Dayana
    ├── run_keypoint_training.sh   ← entrena modelos SFP/SC de keypoints
    ├── predict_keypoints.py       ← inferencia/debug de checkpoints de keypoints
    ├── prepare_checkpoint.py      ← preprocesa checkpoint (smart init, 1 vez)
    ├── run_finetune.sh            ← wrapper que corre los 2 pasos del fine-tune
    └── train_act.py               ← DEPRECATED (solo redirige a run_finetune.sh)
```

Pegar esta carpeta en la raíz del repo `intrinsic-dev/aic`. Los paths son relativos al root.

---

## Setup (una sola vez)

1. **Pegar deps** del `pixi_deps.toml` en el `pixi.toml` del repo, luego:
   ```bash
   pixi install
   ```

2. **Inputs que deben estar listos antes de las 7am:**
   - `data/bags/` — 300 bags CheatCode de Dayana
   - `data/bags_teleop/` — ~60 bags teleop de Diego
   - `data/calibration/tf_static.csv` — ya existe
   - **Dayana integró `extract_keypoints_from_tf()`** ← ver banner arriba

3. **Validar setup** (smoke test del checkpoint, 2 min):
   ```bash
   pixi run python aic_model/load_act.py
   ```
   Debe terminar con `SMOKE TEST OK — el checkpoint se carga correctamente`.

---

## Ejecución (3 comandos)

### Paso 0 — Entrenar Keypoint Estimator / Capa 0

Antes de convertir bags a `LeRobotDataset`, podemos entrenar el modelo visual del diagrama con el Dataset A unificado:

```bash
bash fine_tune_act_v2/fine_tune_act/scripts/run_keypoint_training.sh
```

Esto entrena dos checkpoints:

```text
outputs/keypoint_estimator/sfp/best_sfp.pt
outputs/keypoint_estimator/sc/best_sc.pt
```

Detalles y comandos de prueba: ver `KEYPOINT_ESTIMATOR.md`.

### Comando 1 — Convertir bags a LeRobotDataset (~30-45 min)

```bash
pixi run python scripts/bag_to_lerobot.py \
  --bags_dir data/bags \
  --teleop_dir data/bags_teleop \
  --tf_static data/calibration/tf_static.csv \
  --output data/dataset_lerobot \
  --repo_id aic_team/sfp_sc_insertion \
  --fps 30
```

Si Dayana NO integró su función, el script aborta inmediatamente con un mensaje claro.

### Comando 2 — Fine-tune completo (~5-6h)

```bash
bash scripts/run_finetune.sh
```

Internamente hace:
1. `prepare_checkpoint.py` — descarga checkpoint Aloha de HF Hub, aplica smart partial init, guarda en `outputs/init_smart_ckpt/` (~3 min)
2. `lerobot-train` — CLI oficial de LeRobot, apuntando al checkpoint pre-inicializado (~5-6h)

Output: `outputs/act_aic_run1/checkpoints/last/`. Checkpoints intermedios cada 5000 steps (~25 min) para tener un seguro contra crash.

### Comando 3 — Copiar al modelo final

```bash
mkdir -p models/act_policy
cp outputs/act_aic_run1/checkpoints/last/* models/act_policy/
```

Listo. `models/act_policy/` queda en el formato que `AICPolicy.from_pretrained()` espera.

---

## Qué hace el smart partial init

Las 2 capas con dimensiones distintas entre Aloha (14D state/action) y AIC (32D state, 6D action) reciben tratamiento especial en vez de init random:

| Capa | Aloha shape | AIC shape | Tratamiento |
|------|-------------|-----------|-------------|
| `encoder_robot_state_input_proj.weight` | [512, 14] | [512, 32] | Copia primeras 6 cols + Xavier para las 26 restantes |
| `encoder_robot_state_input_proj.bias` | [512] | [512] | Copia íntegra |
| `action_head.weight` | [14, 512] | [6, 512] | Init pequeño (std=0.01) para evitar explosión de loss |
| `action_head.bias` | [14] | [6] | Init en cero |

Todo lo demás (backbone visual ResNet-18, transformer encoder/decoder, VAE) transfiere normalmente.

**Flag de seguridad:** si algo sale mal, en `prepare_checkpoint.py` se puede pasar `smart_init=False` para volver al comportamiento de init random clásico.

---

## Troubleshooting

| Síntoma | Causa probable | Fix |
|---------|----------------|-----|
| `bag_to_lerobot.py` aborta con "extract_keypoints_from_tf aún es el placeholder" | Dayana no integró | Integrar y cambiar el flag |
| `CUDA out of memory` | batch_size grande | Bajar a 4 en `configs/act_aic.yaml` |
| `policy.save_pretrained` falla | Versión vieja de lerobot | El script ya cae a guardado manual |
| Loss no baja después de 5k steps | Learning rate alto | Bajar `lr` a `5.0e-6` |
| `404 Client Error` al descargar checkpoint | Sin internet | Reintentar; el checkpoint queda cachea
