import oracledb
import os
from dotenv import load_dotenv

load_dotenv()


def connect():
    """
    Realiza conexão com o banco Oracle usando o driver oracledb em modo Thin.
    Não é necessário instalar Oracle Client.
    """

    user = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    host = os.getenv("ORACLE_HOST")
    port = os.getenv("ORACLE_PORT")
    service = os.getenv("ORACLE_SERVICE_NAME")

    dsn = f"{host}:{port}/{service}"

    try:
        connection = oracledb.connect(user=user, password=password, dsn=dsn)
        print("✅ Conectado ao Oracle com sucesso!")
        return connection

    except Exception as e:
        print("❌ Erro ao conectar ao Oracle:", e)
        raise