#!/bin/bash

# Configurações
KEY_DIR="C:/Users/gabri/.ssh"  # Diretório das chaves
PRIVATE_KEY="$KEY_DIR/id_rsa"  # Caminho da chave privada
PUBLIC_KEY="$PRIVATE_KEY.pub"  # Caminho da chave pública

REMOTE_HOST="172.16.9.105"  # Endereço do servidor remoto (fogcomputing)
REMOTE_USER="fogcomputing"  # Usuário no servidor remoto
LOCAL_MODEL_PATH="C:/Users/gabri/Área de Trabalho/tcc/tcc2-FL/pretrained_model_datasetfinalgf.keras"  # Caminho local do arquivo
REMOTE_MODEL_PATH="/home/fogcomputing/Downloads/flower-24/scripts/models"  # Caminho remoto do arquivo

# Gera a chave SSH se não existir
generate_ssh_key() {
    if [ ! -f "$PRIVATE_KEY" ]; then
        echo "Gerando chaves SSH..."
        ssh-keygen -t rsa -b 2048 -f "$PRIVATE_KEY" -N ""
        echo "Chaves SSH geradas com sucesso!"
    else
        echo "Chaves SSH já existem. Pulando geração."
    fi
}

# Adiciona a chave pública ao servidor manualmente
add_public_key_to_server() {
    echo "Adicionando chave pública ao servidor remoto..."
    ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
    scp "$PUBLIC_KEY" "$REMOTE_USER@$REMOTE_HOST:~/.ssh/temp_key.pub"
    ssh "$REMOTE_USER@$REMOTE_HOST" "cat ~/.ssh/temp_key.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && rm ~/.ssh/temp_key.pub"

    if [ $? -eq 0 ]; then
        echo "Chave pública adicionada com sucesso!"
    else
        echo "Erro ao adicionar chave pública."
        exit 1
    fi
}

# Transfere o arquivo via SFTP
transfer_model_sftp() {
    echo "Transferindo arquivo via SFTP..."
    sftp -i "$PRIVATE_KEY" -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" <<EOF
put "$LOCAL_MODEL_PATH" "$REMOTE_MODEL_PATH"
exit
EOF

    if [ $? -eq 0 ]; then
        echo "Transferência concluída com sucesso!"
    else
        echo "Erro ao transferir arquivo."
        exit 1
    fi
}

# Execução principal
generate_ssh_key
add_public_key_to_server
transfer_model_sftp
