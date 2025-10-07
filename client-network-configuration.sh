#!/bin/bash

# List of Pi clients
CLIENTS=(
  "rp-projection-01@192.168.10.101"
  "rp-projection-02@192.168.10.102"
  "rp-projection-03@192.168.10.103"
  "rp-projection-04@192.168.10.104"
)

SERVICE_NAME="pi-projection-client.service"

for CLIENT in "${CLIENTS[@]}"; do
  echo "Configuring $CLIENT..."

  ssh "$CLIENT" "bash -s" <<'EOF'
USER_HOME=\$HOME
SERVICE_PATH=/etc/systemd/system/pi-projection-client.service

cat <<EOL | sudo tee \$SERVICE_PATH
[Unit]
Description=Pi Projection Client
After=network.target

[Service]
Type=simple
WorkingDirectory=\$USER_HOME/pi-projection-client
ExecStart=/bin/bash -c \"source \$USER_HOME/pi-projection-client/init-env.sh && python \$USER_HOME/pi-projection-client/client.py\"
Restart=always
User=\$(whoami)

[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl start "$SERVICE_NAME"
EOF

  echo "$CLIENT configured."
done
echo "All clients configured."
# This script connects to each Raspberry Pi client via SSH,
# sets up a systemd service to run the client application on boot,
# and starts the service immediately.
# Ensure you have SSH access to each client and that the client application
# is located in the specified directory on each client.
# Adjust the CLIENTS array and paths as necessary for your setup.