from ..config.config import connect
import oracledb


class DbSender:
    def __init__(self):
        self.conn = connect()

    def get_or_create_patio(self, patio):
        cursor = self.conn.cursor()

        cursor.execute("SELECT ID FROM PATIO WHERE NOME = :nome", {"nome": patio})
        row = cursor.fetchone()
        if row:
            return row[0]

        id_out = cursor.var(int)
        cursor.execute(
            "INSERT INTO PATIO (NOME) VALUES (:nome) RETURNING ID INTO :id_out",
            {"nome": patio, "id_out": id_out},
        )
        self.conn.commit()
        return id_out.getvalue()

    def get_or_create_setor(self, setor, patio_id):
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT ID FROM SETOR WHERE COR = :cor AND PATIO_ID = :patio_id
        """,
            {"cor": setor, "patio_id": patio_id},
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        id_out = cursor.var(int)
        cursor.execute(
            """
            INSERT INTO SETOR (COR, PATIO_ID)
            VALUES (:cor, :patio_id) RETURNING ID INTO :id_out
        """,
            {"cor": setor, "patio_id": patio_id, "id_out": id_out},
        )
        self.conn.commit()
        return id_out.getvalue()

    def send_motos(self, motos):
        cursor = self.conn.cursor()

        for moto in motos:
            patio_id = self.get_or_create_patio(moto["patio"])
            setor_id = self.get_or_create_setor(moto["setor"], patio_id)

            cursor.execute(
                """
                INSERT INTO MOTO (PLACA, SETOR_ID, CONFIANCA)
                VALUES (:placa, :setor_id, :confianca)
            """,
                {
                    "placa": moto["moto_id"],
                    "setor_id": setor_id,
                    "confianca": float(moto["confianca"]),
                },
            )

        self.conn.commit()
        print("✅ Motos registradas com sucesso!")

    def set_motos(self, motos):
        self.motos = motos