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

```bash
# En WSL (no PowerShell)
cd /mnt/c/Users/diegu/Documents/mis_proyectos/robotica/aic

# Autorizar conexiones X11 desde Docker
xhost +local:docker

# Bajar contenedores si estaban corriendo
docker compose -f docker/docker-compose.yaml down

# Levantar en background
docker compose -f docker/docker-compose.yaml up -d
```

> **Nota:** `xhost +local:docker` debe ejecutarse en cada sesión de WSL nueva,
> antes de `docker compose up`.

---

## Resumen de archivos modificados

| Archivo | Cambio |
|---|---|
| `docker/aic_model/Dockerfile` | Loop de reintentos para `pixi install` + `sed` CRLF fix |
| `docker/docker-compose.yaml` | `DISPLAY` env var + volumen `/tmp/.X11-unix` para Gazebo en WSL |
