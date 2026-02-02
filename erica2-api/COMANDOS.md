# ERICA API - Comandos y Herramientas de Desarrollo

## Dominio Principal
- **Producción**: https://erica.ivf20.app
- **Staging**: https://erica.ivf20.app/staging
- **Development**: http://localhost:8001

---

## 🚀 Inicio Rápido

```bash
# 1. Configurar entorno
export ERICA_ENV=development

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar servidor
python dev_cli.py start
# o
uvicorn main:app --reload --port 8001
```

---

## 📁 Archivos de Entorno

| Archivo | Entorno | Puerto | URL |
|---------|---------|--------|-----|
| `.env.development` | Desarrollo | 8001 | localhost:8001 |
| `.env.staging` | Staging | 8002 | erica.ivf20.app/staging |
| `.env.production` | Producción | 8000 | erica.ivf20.app |

```bash
# Copiar ejemplo
cp .env.example .env.development

# Editar credenciales
nano .env.development
```

---

## 🛠️ CLI Principal (dev_cli.py)

### Entorno
```bash
# Ver configuración actual
python dev_cli.py env show

# Validar configuración
python dev_cli.py env validate

# Cambiar entorno
python dev_cli.py env set staging
```

### Health Checks
```bash
# Local
python dev_cli.py health local

# Remoto
python dev_cli.py health remote --env production
```

### Modelos
```bash
# Listar modelos
python dev_cli.py models list

# Probar carga
python dev_cli.py models test

# Benchmark
python dev_cli.py models benchmark
```

### Ranking
```bash
# Local con imagen
python dev_cli.py rank local /path/to/image.jpg

# Batch (carpeta)
python dev_cli.py rank batch /path/to/folder/

# Remoto
python dev_cli.py rank remote CYCLE_OBJECT_ID --env development
```

### Servidor
```bash
# Iniciar desarrollo
python dev_cli.py start

# Iniciar producción
python dev_cli.py start --env production

# Con PM2
python dev_cli.py start --env production --pm2
```

### Logs
```bash
# Ver logs
python dev_cli.py logs show -n 100

# Tail en tiempo real
python dev_cli.py logs tail

# Limpiar logs
python dev_cli.py logs clear
```

### Requirements y Conda
```bash
# Actualizar requirements.txt
python dev_cli.py requirements

# Exportar entorno conda
python dev_cli.py conda
```

---

## 🔧 Herramientas Individuales

### auto_requirements.py
```bash
# Analizar y actualizar requirements.txt
python auto_requirements.py

# Solo verificar
python auto_requirements.py --check

# Ver imports encontrados
python auto_requirements.py --scan

# Exportar entorno conda/miniconda
python auto_requirements.py --conda
```

**Nota**: Si usas conda/miniconda, el archivo `miniconda_requirements.yml` y `miniconda_requirements.txt` se generarán automáticamente al iniciar la API.

### model_tester.py
```bash
# Todas las pruebas
python model_tester.py test

# Benchmark
python model_tester.py benchmark --iterations 20

# Uso de memoria
python model_tester.py memory

# Verificar archivos
python model_tester.py files
```

### image_selector.py
```bash
# Modo interactivo
python image_selector.py

# Procesar carpeta
python image_selector.py --folder ./images --age 32

# Procesar imagen
python image_selector.py --image embryo.jpg
```

### deploy.py
```bash
# Ver estado
python deploy.py status

# Iniciar
python deploy.py start development
python deploy.py start staging
python deploy.py start production

# Detener
python deploy.py stop development

# Reiniciar
python deploy.py restart production

# Ver logs
python deploy.py logs production -n 100

# Deploy completo
python deploy.py deploy staging

# Crear ecosystem.config.js
python deploy.py ecosystem
```

### version_manager.py
```bash
# Ver versión actual
python version_manager.py show

# Bump versión
python version_manager.py bump patch    # 2.0.0 -> 2.0.1
python version_manager.py bump minor    # 2.0.0 -> 2.1.0
python version_manager.py bump major    # 2.0.0 -> 3.0.0

# Crear tag
python version_manager.py tag -m "Release 2.1.0"

# Generar changelog
python version_manager.py changelog
```

---

## 📋 Scripts Bash (scripts.sh)

```bash
# Cargar funciones
source scripts.sh
requirements           # Actualizar requirements.txt
erica-conda                  # Exportar conda environment
erica-
# Ver ayuda
erica-help

# Comandos rápidos
erica-dev                    # Cambiar a desarrollo
erica-start                  # Iniciar servidor
erica-health                 # Health check
erica-test                   # Correr tests
erica-logs prod              # Ver logs producción
```

---

## 🌐 API Endpoints

### Health Check (sin autenticación)
```bash
curl https://erica.ivf20.app/health
```

### Root
```bash
curl https://erica.ivf20.app/
```

### Status (requiere API Key)
```bash
curl https://erica.ivf20.app/status \
  -H "X-API-Key: YOUR_API_KEY"
```

### Ranking (requiere 2 llaves de seguridad)
```bash
# Producción
curl -X POST https://erica.ivf20.app/rankthisone \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_SECRET_KEY" \
  -d '{
    "objectId": "CYCLE_OBJECT_ID",
    "validation_key": "YOUR_VALIDATION_PASS_KEY"
  }'

# Desarrollo (localhost)
curl -X POST http://localhost:8001/rankthisone \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev_secret_key_12345" \
  -d '{
    "objectId": "CYCLE_OBJECT_ID",
    "validation_key": "dev_validation_key_12345"
  }'
```

**⚠️ Llaves Requeridas:**
1. **`X-API-Key`** (Header) - Autenticación de la API
   - Variable: `API_SECRET_KEY` en `.env.*`
   - Ubicación: Header HTTP
   
2. **`validation_key`** (Body) - Validación de operación de ranking
   - Variable: `VALIDATION_PASS_KEY` en `.env.*`
   - Ubicación: Body del request JSON

---

## 🔐 Seguridad

### Llaves de Seguridad (configurar en .env.*)

| Variable | Ubicación | Descripción | Uso |
|----------|-----------|-------------|-----|
| `API_SECRET_KEY` | Header `X-API-Key` | Autenticación de la API | Todos los endpoints protegidos |
| `VALIDATION_PASS_KEY` | Body `validation_key` | Validación de ranking | Solo endpoint `/rankthisone` |

**Ejemplo de configuración en `.env.development`:**
```bash
API_SECRET_KEY=dev_secret_key_12345
VALIDATION_PASS_KEY=dev_validation_key_12345
```

**Ejemplo de configuración en `.env.production`:**
```bash
API_SECRET_KEY=PROD_SECURE_KEY_CHANGE_ME
VALIDATION_PASS_KEY=PROD_VALIDATION_KEY_CHANGE_ME
```

### Headers Requeridos
| Header | Valor | Cuándo |
|--------|-------|--------|
| `X-API-Key` | Valor de `API_SECRET_KEY` | Endpoints protegidos |
| `Content-Type` | `application/json` | Requests con body |

### Endpoints Protegidos
- `POST /rankthisone` - Requiere `X-API-Key` (header) + `validation_key` (body)
- `GET /status` - Requiere `X-API-Key` (header)
- `GET /debug/*` - Solo en desarrollo, requiere `X-API-Key` (header)

### Endpoints Públicos
- `GET /` - Root info (sin autenticación)
- `GET /health` - Health check (sin autenticación)

---

## 📝 Ejemplos de Llamadas Completas

### Health Check (sin autenticación)
```bash
# Local
curl http://localhost:8001/health

# Producción
curl https://erica.ivf20.app/health
```

### Status (requiere API Key)
```bash
# Local
curl http://localhost:8001/status \
  -H "X-API-Key: dev_secret_key_12345"

# Producción
curl https://erica.ivf20.app/status \
  -H "X-API-Key: TU_API_SECRET_KEY"
```

### Ranking (requiere ambas llaves)
```bash
# Local - Development
curl -X POST http://localhost:8001/rankthisone \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev_secret_key_12345" \
  -d '{
    "objectId": "ABC123XYZ",
    "validation_key": "dev_validation_key_12345"
  }'

# Producción
curl -X POST https://erica.ivf20.app/rankthisone \
  -H "Content-Type: application/json" \
  -H "X-API-Key: TU_API_SECRET_KEY" \
  -d '{
    "objectId": "ABC123XYZ",
    "validation_key": "TU_VALIDATION_PASS_KEY"
  }'
```

### Desde JavaScript/TypeScript
```javascript
// Ejemplo de llamada desde frontend
const rankCycle = async (objectId) => {
  const response = await fetch('https://erica.ivf20.app/rankthisone', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': process.env.API_SECRET_KEY // Tu API key
    },
    body: JSON.stringify({
      objectId: objectId,
      validation_key: process.env.VALIDATION_PASS_KEY // Tu validation key
    })
  });
  
  return await response.json();
};
```

### Desde Python
```python
import requests
import os

def rank_cycle(object_id):
    url = "https://erica.ivf20.app/rankthisone"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": os.getenv("API_SECRET_KEY")
    }
    payload = {
        "objectId": object_id,
        "validation_key": os.getenv("VALIDATION_PASS_KEY")
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()
```

---

## ⚙️ Migrar Llamadas Antiguas

Si tenías llamadas previas sin autenticación, necesitas actualizarlas:

### ❌ Llamada Antigua (sin seguridad)
```bash
curl -X POST https://erica.ivf20.app/rankthisone \
  -H "Content-Type: application/json" \
  -d '{"objectId": "ABC123"}'
```

### ✅ Llamada Nueva (con seguridad)
```bash
curl -X POST https://erica.ivf20.app/rankthisone \
  -H "Content-Type: application/json" \
  -H "X-API-Key: TU_API_SECRET_KEY" \
  -d '{
    "objectId": "ABC123",
    "validation_key": "TU_VALIDATION_PASS_KEY"
  }'
```

### Cambios necesarios:
1. ✅ Agregar header `X-API-Key` con el valor de `API_SECRET_KEY`
2. ✅ Agregar campo `validation_key` en el body con el valor de `VALIDATION_PASS_KEY`
3. ✅ Configurar ambas variables en tu archivo `.env.*`

---

## 🐳 Docker

```bash
# Build
docker build -t erica-api .

# Run desarrollo
docker run -p 8001:8001 -e ERICA_ENV=development erica-api

# Run producción
docker run -p 8000:8000 -e ERICA_ENV=production erica-api

# Docker Compose
docker-compose up -d
```

---

## 📊 PM2 Commands

```bash
# Estado
pm2 status

# Iniciar todo
pm2 start ecosystem.config.js

# Logs
pm2 logs erica-prod

# Monitoreo
pm2 monit

# Reiniciar
pm2 restart erica-prod

# Detener
pm2 stop erica-prod

# Eliminar
pm2 delete erica-prod
```

---

## 🗄️ Estructura de Carpetas

```
api/erica-api/
├── main.py              # Entry point FastAPI
├── erica_api.py         # Pipeline principal
├── config.py            # Configuración centralizada
├── dev_cli.py           # CLI desarrollo
├── auto_requirements.py # Auto-sync requirements
├── model_tester.py      # Testing modelos
├── image_selector.py    # Selector imágenes
├── deploy.py            # Deployment manager
├── version_manager.py   # Versionado
├── scripts.sh           # Scripts bash
├── .env.development     # Config desarrollo
├── .env.staging         # Config staging
├── .env.production      # Config producción
├── requirements.txt     # Dependencias
├── models/              # Modelos ML
│   ├── erica_model2.pth
│   ├── erica_cropper.pt
│   ├── erica_segmentor_n.pt
│   └── scaler_info.json
├── utils/               # Utilidades
│   ├── erica_pipeline.py
│   ├── erica_cropper.py
│   ├── erica_model.py
│   └── ...
└── logs/                # Logs
```

---

## 🧪 Testing Local

```bash
# 1. Iniciar servidor desarrollo
python dev_cli.py start

# 2. En otra terminal, probar health
curl http://localhost:8001/health

# 3. Probar ranking local
python image_selector.py --image test_embryo.jpg

# 4. Probar ranking remoto (necesitas objectId válido)
python dev_cli.py rank remote ABC123 --env development
```

---

## 🔄 Workflow Típico

### Desarrollo
1. `export ERICA_ENV=development`
2. `python dev_cli.py start`
3. Hacer cambios
4. Probar con `curl` o `image_selector.py`
5. `python dev_cli.py logs tail`

### Deploy a Staging
1. `git commit -am "feat: nueva funcionalidad"`
2. `python version_manager.py bump minor`
3. `python deploy.py deploy staging`
4. `python dev_cli.py health remote --env staging`

### Deploy a Producción
1. `git merge staging`
2. `python version_manager.py bump release`
3. `python version_manager.py tag`
4. `python deploy.py deploy production`
5. `python dev_cli.py health remote --env production`
