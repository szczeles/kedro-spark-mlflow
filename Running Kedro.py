# Databricks notebook source
# MAGIC %pip install -r src/requirements-databricks.txt

# COMMAND ----------

import logging
from pathlib import Path

# suppress excessive logging from py4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# copy project data into DBFS
project_root = Path.cwd()
data_dir = project_root / "data" / "01_raw"
dbutils.fs.cp(
    f"file://{data_dir.as_posix()}", f"dbfs:///data/01_raw", recurse=True
)

# make sure the data has been copied
dbutils.fs.ls("data/01_raw")

# COMMAND ----------

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path

bootstrap_project(Path.cwd())

with KedroSession.create(project_path=Path.cwd()) as session:
    session.run()

# COMMAND ----------

# MAGIC %pip install kedro-viz

# COMMAND ----------

# MAGIC %run_viz

# COMMAND ----------

# MAGIC %reload_kedro

# COMMAND ----------


