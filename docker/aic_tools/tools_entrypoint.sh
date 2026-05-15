#!/bin/bash
# Entrypoint del contenedor de herramientas ROS 2.
# Fuente el workspace compilado y deja el contenedor vivo para docker exec.
set -e

source /ws_aic/install/setup.bash

# Disponibilidad automática en sesiones interactivas (docker exec -it ... bash)
grep -qxF 'source /ws_aic/install/setup.bash' /root/.bashrc \
  || echo 'source /ws_aic/install/setup.bash' >> /root/.bashrc

exec tail -f /dev/null
