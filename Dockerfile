# Utilizar una imagen base oficial de Python
FROM python:3.8

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias y instalarlas
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación al contenedor
COPY . .

# Comando para ejecutar la aplicación usando uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
