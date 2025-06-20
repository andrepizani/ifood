# Databricks notebook source
# MAGIC %md
# MAGIC #Imports
# MAGIC

# COMMAND ----------

import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, countDistinct, count, sum, avg, when, round, lit, lag, datediff, dayofweek,
    concat_ws, floor
)
from pyspark.sql.window import Window


# COMMAND ----------

# MAGIC %md
# MAGIC #FILES

# COMMAND ----------

# URLs dos arquivos
urls = {
    "order.json.gz": "https://data-architect-test-source.s3-sa-east-1.amazonaws.com/order.json.gz",
    "consumer.csv.gz": "https://data-architect-test-source.s3-sa-east-1.amazonaws.com/consumer.csv.gz",
    "restaurant.csv.gz": "https://data-architect-test-source.s3-sa-east-1.amazonaws.com/restaurant.csv.gz",
    "ab_test_ref.tar.gz": "https://data-architect-test-source.s3-sa-east-1.amazonaws.com/ab_test_ref.tar.gz"
}

# Pasta temporária para download
tmp_path = "/tmp"
dbfs_path = "dbfs:/FileStore/tables"


# COMMAND ----------

# MAGIC %md
# MAGIC #Functions

# COMMAND ----------

def extract_csv(dbfs_tar_gz_path, dbfs_csv_output_path):
    local_tar = "/tmp/temp.tar.gz"
    local_extract = "/tmp/extract_csv"

    dbutils.fs.cp(dbfs_tar_gz_path, f"file:{local_tar}", True)
    os.makedirs(local_extract, exist_ok=True)

    with tarfile.open(local_tar, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".csv") and not member.name.startswith("._"):
                tar.extract(member, path=local_extract)
                break  


    csv_file = [f for f in os.listdir(local_extract) if f.endswith(".csv")][0]
    local_csv_path = os.path.join(local_extract, csv_file)
    df = spark.read.csv(f"file:{local_csv_path}", header=True, inferSchema=True)
    df.write.mode("overwrite").option("header", True).csv(dbfs_csv_output_path)

    return df
    
def download_para_dbfs(nome_arquivo, url):
    local_tmp = os.path.join(tmp_path, nome_arquivo)
    urllib.request.urlretrieve(url, local_tmp)
    dbutils.fs.cp(f"file:{local_tmp}", f"{dbfs_path}/{nome_arquivo}")


# COMMAND ----------

# MAGIC %md
# MAGIC # Download Arquivos

# COMMAND ----------

for nome, url in urls.items():
    print(f"Download: {nome}...")
    download_para_dbfs(nome, url)

# COMMAND ----------

# MAGIC %md
# MAGIC # Ler arquivos
# MAGIC

# COMMAND ----------

df_consumer = spark.read.csv("dbfs:/FileStore/tables/consumer.csv.gz", header=True, inferSchema=True)
df_restaurant = spark.read.csv("dbfs:/FileStore/tables/restaurant.csv.gz", header=True, inferSchema=True)
df_ab_test = extract_csv("dbfs:/FileStore/tables/ab_test_ref.tar.gz","dbfs:/FileStore/tables/ab_test.ref")
df_order = spark.read.json("dbfs:/FileStore/tables/order.json.gz")


# # JOIN da base de pedidos com teste A/B
# df = df_order.join(df_ab_test, on="customer_id", how="inner") \
#                   .join(df_consumer, on="customer_id", how="left") \
#                   .join(df_restaurant, df_order.merchant_id == df_restaurant.id, how="left")



# COMMAND ----------


# 1. Contar a quantidade de usuários por grupo (antes da amostragem)
df_ab_test.groupBy("is_target").agg(
    F.countDistinct("customer_id").alias("usuarios_unicos")
).show()

# 2. Contar tamanho do grupo menor
min_count = df_ab_test.groupBy("is_target").agg(
    F.countDistinct("customer_id").alias("n")
).agg(F.min("n")).first()[0]

# 3. Amostrar igual para os dois grupos com applyInPandas
df_ab_test = df_ab_test.groupBy("is_target").applyInPandas(
    lambda pdf: pdf.sample(n=min_count, random_state=25),
    schema=df_ab_test.schema
)



df_ab_order = df_order.join(df_ab_test, on="customer_id", how="inner")

df_ab_order.groupBy("is_target").agg(
    F.countDistinct("customer_id").alias("usuarios_unicos")
).show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Analise Exploratória

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificando dados nulos no df_order

# COMMAND ----------

def null_report_long(df, name):
    """
    Mostra, em formato coluna-valor, quantos nulos há em cada coluna de `df`.
    """
    total = df.count()                            

    # 1) Conta nulos por coluna (uma única linha, várias colunas)
    counts_row = df.agg(*[
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
    ])

    # 2) Converte para formato longo com stack: column | nulls | pct_nulls
    expr_list  = ", ".join([f"'{c}', {c}" for c in df.columns])
    stack_expr = f"stack({len(df.columns)}, {expr_list}) as (column, nulls)"
    
    long_df = (
        counts_row.selectExpr(stack_expr)
                  .withColumn("pct_nulls",
                              F.round(F.col("nulls") * 100.0 / total, 2))
                  .orderBy(F.desc("nulls"))
    )

    print(f"=== Nulos em {name} (total linhas = {total}) ===")
    display(long_df)   # tabela interativa no Databricks
null_report_long(df_ab_order, "order")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise prévia dos totais

# COMMAND ----------

df_ab_order.select(
    F.min("order_total_amount").alias("minimo"),
    F.mean("order_total_amount").alias("media"),
    # Mediana aproximada
    F.expr("percentile_approx(order_total_amount, 0.5)").alias("mediana"),
    # Percentil 95 aproximado
    F.expr("percentile_approx(order_total_amount, 0.95)").alias("p95"),
    F.max("order_total_amount").alias("maximo")
).show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Verificando quantidades de pedidos entre faixa de valores

# COMMAND ----------


print("=== Analisar Faixas de Valor ===")

# Calcular início da faixa
df_faixas = df_ab_order.withColumn("inicio_faixa", floor(col("order_total_amount") / 100) * 100)

# Criar rótulo 
df_faixas = df_faixas.withColumn(
    "faixa_valor",
    concat_ws("–", col("inicio_faixa"), (col("inicio_faixa") + 99))
)

# Agrupar por faixa e ordenar corretamente pela faixa numérica
df_tabela_faixas = df_faixas.groupBy("faixa_valor", "inicio_faixa") \
                            .count() \
                            .orderBy("inicio_faixa")

# Exibir apenas faixa_valor e count
df_tabela_faixas.select("faixa_valor", "count").display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Removendo outliers

# COMMAND ----------

df_ab_order.select(
    F.min("order_total_amount").alias("minimo"),
    F.mean("order_total_amount").alias("media"),
    # Mediana aproximada
    F.expr("percentile_approx(order_total_amount, 0.5)").alias("mediana"),
    # Percentil 95 aproximado
    F.expr("percentile_approx(order_total_amount, 0.95)").alias("p95"),
    F.max("order_total_amount").alias("maximo")
).show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Analise dos indicadores

# COMMAND ----------



# ------------------ Parte 1: KPIs por Grupo ------------------
df_kpis_grupo = df_ab_order.groupBy("is_target").agg(
    countDistinct("customer_id").alias("usuarios_unicos"),
    count("order_id").alias("total_pedidos"),
    sum("order_total_amount").alias("gasto_total"),
    round(sum("order_total_amount") / count("order_id"), 2).alias("ticket_medio")
)

# ------------------ Parte 2: KPIs por Usuário ------------------
df_usuario = df_ab_order.groupBy("customer_id", "is_target").agg(
    count("order_id").alias("qtd_pedidos"),
    sum("order_total_amount").alias("gasto_total")
)

df_usuario = df_usuario.withColumn(
    "recorrente", when(col("qtd_pedidos") > 1, 1).otherwise(0)
)

df_agregado_usuario = df_usuario.groupBy("is_target").agg(
    round((sum("recorrente") / count("customer_id")) * 100, 2).alias("percentual_recorrentes"),
    round(avg("qtd_pedidos"), 2).alias("pedidos_por_usuario"),
    round(avg("gasto_total"), 2).alias("ticket_medio_por_usuario")
)

# ------------------ Parte 3: Tempo médio entre pedidos ------------------
janela = Window.partitionBy("customer_id").orderBy("order_created_at")

df_lag = df_ab_order.select("customer_id", "is_target", "order_created_at") \
    .withColumn("pedido_anterior", lag("order_created_at").over(janela)) \
    .withColumn("delta_dias", datediff("order_created_at", "pedido_anterior"))

df_tempo = df_lag.groupBy("is_target").agg(
    round(avg("delta_dias"), 2).alias("tempo_medio_entre_pedidos_dias")
)

# ------------------ Parte 4: Fim de semana ------------------
df_fds = df_ab_order.withColumn("dia_semana", dayofweek("order_created_at")) \
    .withColumn("eh_fds", when(col("dia_semana").isin([1, 6, 7]), 1).otherwise(0))

df_fds_agregado = df_fds.groupBy("is_target").agg(
    sum("eh_fds").alias("pedidos_fds"),
    round((sum("eh_fds") / count("order_id")) * 100, 2).alias("percentual_fds")
)

# ------------------ Parte Final: Join de tudo ------------------
df_final = df_kpis_grupo \
    .join(df_agregado_usuario, on="is_target", how="inner") \
    .join(df_tempo, on="is_target", how="inner") \
    .join(df_fds_agregado, on="is_target", how="inner")

# Mostrar resultado final
df_final.show(truncate=False)


# COMMAND ----------


# 1. Agregações básicas por grupo
df_grupo = df_ab_order.groupBy("is_target").agg(
    countDistinct("customer_id").alias("n_usuarios_com_pedido"),
    count("order_id").alias("n_pedidos"),
    avg("order_total_amount").alias("ticket_medio"),
    sum("order_total_amount").alias("valor_total")
)

# 2. Coletar totais para cálculo de proporções
totais = df_grupo.agg(
    sum("n_usuarios_com_pedido").alias("total_usuarios"),
    sum("n_pedidos").alias("total_pedidos")
).first()

total_usuarios = totais["total_usuarios"]
total_pedidos = totais["total_pedidos"]

print("=== Totais Gerais ===")
print(f"Total de usuários com pedido: {total_usuarios}")
print(f"Total de pedidos: {total_pedidos}")

# 3. Adicionar colunas de proporção (%)
df_grupo = df_grupo.withColumn(
    "pct_usuarios_com_pedido", round(col("n_usuarios_com_pedido") * 100 / total_usuarios, 2)
).withColumn(
    "pct_pedidos", round(col("n_pedidos") * 100 / total_pedidos, 2)
)

# 4. Exibir resultado final
df_grupo.select(
    "is_target",
    "n_usuarios_com_pedido",
    "pct_usuarios_com_pedido",
    "n_pedidos",
    "pct_pedidos",
    "ticket_medio",
    "valor_total"
).show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Analise por dispositivo

# COMMAND ----------


# Primeiro, filtrar outliers e considerar apenas quem fez pedido
df_pedidos = df_ab_order.select("customer_id", "origin_platform", "is_target", "order_id", "order_total_amount")

# Agrupar por plataforma e grupo
df_por_usuario = df_pedidos.groupBy("origin_platform", "is_target").agg(
    count("order_id").alias("n_pedidos"),
    countDistinct("customer_id").alias("n_usuarios_com_pedido"),
    avg("order_total_amount").alias("ticket_medio")
)

# Calcular pedidos por usuário

df_por_usuario = df_por_usuario.withColumn(
    "pedidos_por_usuario", round(col("n_pedidos") / col("n_usuarios_com_pedido"), 2)
).orderBy("origin_platform", "is_target")

df_por_usuario.select(
    "origin_platform", "is_target", "n_usuarios_com_pedido", "n_pedidos", "pedidos_por_usuario", "ticket_medio"
).show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizacão de dados e testagem de dados

# COMMAND ----------

from pyspark.sql.functions import col, hour, avg, floor, lit, concat_ws, date_format
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# COMMAND ----------

df_plot = df_por_usuario.select(
    "origin_platform", "is_target", "pedidos_por_usuario", "ticket_medio"
).toPandas()

sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Pedidos por usuário
sns.barplot(
    data=df_plot,
    x="origin_platform",
    y="pedidos_por_usuario",
    hue="is_target",
    ax=axes[0]
)
axes[0].set_title("Pedidos por Usuário por Plataforma")
axes[0].set_xlabel("Plataforma")
axes[0].set_ylabel("Pedidos por Usuário")
axes[0].legend(title="Grupo")

# Gráfico 2: Ticket médio
sns.barplot(
    data=df_plot,
    x="origin_platform",
    y="ticket_medio",
    hue="is_target",
    ax=axes[1]
)
axes[1].set_title("Ticket Médio por Plataforma")
axes[1].set_xlabel("Plataforma")
axes[1].set_ylabel("Ticket Médio (R$)")
axes[1].legend(title="Grupo")

# Ajustar layout
plt.tight_layout()
plt.show()


# COMMAND ----------


# ------------------ 1. Ticket médio por dia da semana ------------------
df_dia_semana = df_ab_order.withColumn("dia_semana", date_format("order_created_at", "E")) \
    .groupBy("dia_semana") \
    .agg(avg("order_total_amount").alias("ticket_medio")) \
    .toPandas()

# Ordenar os dias corretamente
dias_ordem = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
df_dia_semana["dia_semana"] = pd.Categorical(df_dia_semana["dia_semana"], categories=dias_ordem, ordered=True)
df_dia_semana = df_dia_semana.sort_values("dia_semana")

# ------------------ 2. Ticket médio por faixa de horário ------------------
# Criar faixa de hora: início de cada faixa de 3h
df_ab_order_com_hora = df_ab_order.withColumn("faixa_hora", floor(hour("order_created_at") / 3) * 3)

# Construir faixa de horário como string do tipo "00–03", "03–06"...
df_ab_order_com_hora = df_ab_order_com_hora.withColumn(
    "faixa_horario",
    concat_ws("–",
        col("faixa_hora").cast("string"),
        (col("faixa_hora") + 3).cast("string")
    )
)
df_faixa_hora = df_ab_order_com_hora.groupBy("faixa_horario") \
    .agg(avg("order_total_amount").alias("ticket_medio")) \
    .toPandas().sort_values("faixa_horario")

# ------------------ 3. Plotagem ------------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Gráfico 1: Vertical por dia da semana
sns.barplot(
    data=df_dia_semana,
    x="dia_semana",
    y="ticket_medio",
    ax=axes[0],
    color="#1f77b4"  # Cor única
)
axes[0].set_title("Ticket Médio por Dia da Semana")
axes[0].set_xlabel("Dia da Semana")
axes[0].set_ylabel("Ticket Médio (R$)")

# Gráfico 2: Horizontal por faixa de horário
sns.barplot(
    data=df_faixa_hora,
    y="faixa_horario",
    x="ticket_medio",
    ax=axes[1],
    color="#1f77b4"  # Mesma cor
)
axes[1].set_title("Ticket Médio por Faixa de Horário")
axes[1].set_xlabel("Ticket Médio (R$)")
axes[1].set_ylabel("Faixa de Horário")

plt.tight_layout()
plt.show()


# COMMAND ----------

# Agrupar valor total por dia
df_valor_por_dia = df_ab_order.withColumn("data", to_date("order_created_at")) \
    .groupBy("data") \
    .agg(sum("order_total_amount").alias("valor_total")) \
    .orderBy("data") \
    .toPandas()

# Plotar
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_valor_por_dia, x="data", y="valor_total", color="#1f77b4")
plt.title("Valor Total de Pedidos por Dia")
plt.xlabel("Data")
plt.ylabel("Valor Total (R$)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

# Agrupar valor total por data e grupo
df_valor_por_dia = df_ab_order.withColumn("data", to_date("order_created_at")) \
    .groupBy("data", "is_target") \
    .agg(sum("order_total_amount").alias("valor_total")) \
    .orderBy("data", "is_target") \
    .toPandas()

# Plotar
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_valor_por_dia, x="data", y="valor_total", hue="is_target")
plt.title("Valor Total de Pedidos por Dia (Segmentado por Grupo)")
plt.xlabel("Data")
plt.ylabel("Valor Total (R$)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

# Agrupar por data e calcular valor total e número de pedidos por dia
df_valor_medio_por_dia = df_ab_order.withColumn("data", to_date("order_created_at")) \
    .groupBy("data") \
    .agg(
        sum("order_total_amount").alias("valor_total"),
        count("order_id").alias("qtd_pedidos")
    ) \
    .withColumn("valor_medio_pedido", col("valor_total") / col("qtd_pedidos")) \
    .orderBy("data") \
    .toPandas()

# Plotar valor médio por pedido
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_valor_medio_por_dia, x="data", y="valor_medio_pedido", color="#1f77b4")
plt.title("Valor Médio por Pedido por Dia")
plt.xlabel("Data")
plt.ylabel("Valor Médio (R$)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

# Agrupar valor total e quantidade de pedidos por data e grupo
df_valor_medio_por_dia = df_ab_order.withColumn("data", to_date("order_created_at")) \
    .groupBy("data", "is_target") \
    .agg(
        sum("order_total_amount").alias("valor_total"),
        count("order_id").alias("qtd_pedidos")
    ) \
    .withColumn("valor_medio_pedido", col("valor_total") / col("qtd_pedidos")) \
    .orderBy("data", "is_target") \
    .toPandas()

# Plotar
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_valor_medio_por_dia, x="data", y="valor_medio_pedido", hue="is_target")
plt.title("Valor Médio por Pedido por Dia (Segmentado por Grupo)")
plt.xlabel("Data")
plt.ylabel("Valor Médio (R$)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

# Agrupar quantidade de pedidos por data e grupo
df_qtd_pedidos_por_dia = df_ab_order.withColumn("data", to_date("order_created_at")) \
    .groupBy("data", "is_target") \
    .agg(count("order_id").alias("qtd_pedidos")) \
    .orderBy("data", "is_target") \
    .toPandas()

# Plotar
plt.figure(figsize=(14, 6))
sns.lineplot(data=df_qtd_pedidos_por_dia, x="data", y="qtd_pedidos", hue="is_target")
plt.title("Quantidade de Pedidos por Dia (Segmentado por Grupo)")
plt.xlabel("Data")
plt.ylabel("Número de Pedidos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

df_pizza = df_fds_agregado.select("is_target", "pedidos_fds").toPandas()

# Total de pedidos por grupo (você já tem isso em df_kpis_grupo)
df_total = df_kpis_grupo.select("is_target", "total_pedidos").toPandas()

# Juntar os dois DataFrames
df_merge = pd.merge(df_pizza, df_total, on="is_target")
df_merge["pedidos_dia_util"] = df_merge["total_pedidos"] - df_merge["pedidos_fds"]

# Plotar dois gráficos
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for i, grupo in enumerate(df_merge["is_target"]):
    sizes = [
        df_merge.loc[i, "pedidos_dia_util"],
        df_merge.loc[i, "pedidos_fds"]
    ]
    labels = ["Dias úteis", "Fim de semana"]
    colors = ["#99d8c9", "#fc9272"]

    axes[i].pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    axes[i].set_title(f"Grupo: {grupo.capitalize()}")
    axes[i].axis("equal")  # Deixar o círculo redondo

plt.suptitle("Distribuição de Pedidos por Grupo – Dias Úteis vs. Fim de Semana")
plt.tight_layout()
plt.show()


# COMMAND ----------


# Calcular KPIs por região
df_por_regiao = df_ab_order.groupBy("delivery_address_district").agg(
    count("order_id").alias("qtd_pedidos"),
    sum("order_total_amount").alias("valor_total")
).withColumn(
    "ticket_medio", round(col("valor_total") / col("qtd_pedidos"), 2)
)

# Converter para Pandas
df_por_regiao_pd = df_por_regiao.toPandas()

# Top 10 por ticket médio
top10_ticket = df_por_regiao_pd.sort_values("ticket_medio", ascending=False).head(10)

# Top 10 por quantidade de pedidos
top10_pedidos = df_por_regiao_pd.sort_values("qtd_pedidos", ascending=False).head(10)

# Plotar os dois gráficos lado a lado
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Gráfico 1: Top 10 por ticket médio
sns.barplot(data=top10_ticket, y="delivery_address_district", x="ticket_medio", ax=axes[0])
axes[0].set_title("Top 10 Regiões - Ticket Médio")
axes[0].set_xlabel("Ticket Médio (R$)")
axes[0].set_ylabel("Região")

# Gráfico 2: Top 10 por quantidade de pedidos
sns.barplot(data=top10_pedidos, y="delivery_address_district", x="qtd_pedidos", ax=axes[1])
axes[1].set_title("Top 10 Regiões - Quantidade de Pedidos")
axes[1].set_xlabel("Número de Pedidos")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()


# COMMAND ----------

