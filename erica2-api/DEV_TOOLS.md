# ERICA API - Herramientas de Desarrollo

## 🎯 Descripción General

Sistema completo de herramientas de desarrollo para ERICA API con soporte multi-entorno (development, staging, production).

**Dominio**: https://erica.ivf20.app

---

## 📦 Herramientas Disponibles

### 1. **dev_cli.py** - CLI Principal
CLI completo para desarrollo, testing y deployment.

```bash
python dev_cli.py env show                       # Ver configuración
python dev_cli.py health local                   # Health check local
python dev_cli.py models test                    # Probar modelos
python dev_cli.py rank local img.jpg --age <AGE> # Ranking local con edad madre
python dev_cli.py start                          # Iniciar servidor
python dev_cli.py logs tail                      # Ver logs en tiempo real
python dev_cli.py requirements                   # Actualizar requirements
python dev_cli.py conda                          # Exportar entorno conda
```

### 2. **auto_requirements.py** - Gestión de Dependencias
Sincronización automática de requirements.txt y entorno conda.

```bash
python auto_requirements.py          # Actualizar requirements.txt
python auto_requirements.py --check  # Solo verificar
python auto_requirements.py --conda  # Exportar conda environment
```

**Características**:
- Escanea imports en todos los archivos Python
- Compara con paquetes instalados
- Genera requirements.txt automáticamente
- Detecta y exporta entorno conda/miniconda
- Se ejecuta automáticamente al iniciar la API

**Archivos generados**:
- `requirements.txt` - Dependencias pip
- `miniconda_requirements.yml` - Entorno conda (YAML)
- `miniconda_requirements.txt` - Lista de paquetes conda

### 3. **model_tester.py** - Testing de Modelos
Suite completa de testing para modelos ML.

```bash
python model_tester.py test          # Todas las pruebas
python model_tester.py benchmark     # Benchmarks de rendimiento
python model_tester.py memory        # Uso de memoria
python model_tester.py files         # Verificar archivos
```

**Pruebas incluidas**:
- ✅ Carga de modelos (cropper, segmentor, scoring)
- ✅ Benchmarks de inferencia
- ✅ Análisis de memoria
- ✅ Verificación de archivos

### 4. **image_selector.py** - Selector Interactivo de Imágenes
Herramienta para procesar imágenes localmente.

```bash
python image_selector.py             # Modo interactivo
python image_selector.py --folder ./images
python image_selector.py --image embryo.jpg --age 32
```

**Funciones**:
- 🖼️ Selección interactiva de imágenes
- 🔍 Ranking local sin cloud
- 💾 Guardado de resultados
- 📊 Visualización de scores

### 5. **deploy.py** - Gestión de Deployment
Deployment manager para múltiples entornos.

```bash
python deploy.py status              # Ver estado de todos los entornos
python deploy.py start production    # Iniciar producción
python deploy.py deploy staging      # Deploy completo a staging
python deploy.py logs production     # Ver logs
python deploy.py ecosystem           # Crear PM2 config
```

**Características**:
- 🚀 Deploy automatizado
- 🔄 Health checks post-deploy
- 📊 Monitoreo PM2
- 🐳 Soporte Docker

### 6. **version_manager.py** - Versionado Semántico
Gestión de versiones y releases.

```bash
python version_manager.py show       # Ver versión actual
python version_manager.py bump patch # 2.0.0 -> 2.0.1
python version_manager.py bump minor # 2.0.0 -> 2.1.0
python version_manager.py tag        # Crear git tag
python version_manager.py changelog  # Generar changelog
```

**Funciones**:
- 📌 Versionado semántico
- 🏷️ Git tags automáticos
- 📝 Generación de changelog
- 🔄 Actualización en todos los archivos

### 7. **scripts.sh** - Scripts Bash Rápidos
Atajos para comandos frecuentes.

```bash
source scripts.sh                # Cargar funciones
erica-help                       # Ver ayuda
erica-start                      # Iniciar servidor
erica-conda                      # Exportar conda env
erica-test                       # Correr tests
erica-deploy staging             # Deploy rápido
```

---

## 🌍 Entornos

| Entorno | Puerto | URL | PM2 Name |
|---------|--------|-----|----------|
| Development | 8001 | localhost:8001 | erica-dev |
| Staging | 8002 | erica.ivf20.app/staging | erica-staging |
| Production | 8000 | erica.ivf20.app | erica-prod |

**Archivos de configuración**:
- `.env.development` - Desarrollo local
- `.env.staging` - Pre-producción
- `.env.production` - Producción

**Variables clave**:
```bash
ERICA_ENV=production
API_SECRET_KEY=your_secret_key
VALIDATION_PASS_KEY=your_validation_key
PARSE_SERVER_URL=https://dish-s.ivf20.app/db
```

---

## 🔧 Configuración Inicial

### 1. Configurar entorno
```bash
# Copiar ejemplo
cp .env.example .env.development

# Editar variables
nano .env.development

# Configurar entorno
export ERICA_ENV=development
```

### 2. Instalar dependencias
```bash
# Con pip
pip install -r requirements.txt

# Con conda (si usas miniconda)
conda env create -f miniconda_requirements.yml
# o
conda env update -f miniconda_requirements.yml
```

### 3. Verificar instalación
```bash
python dev_cli.py env validate
python dev_cli.py models test
```

### 4. Iniciar servidor
```bash
python dev_cli.py start
# o
./scripts.sh start
```

---

## 🔄 Workflows Comunes

### Desarrollo Local
```bash
# 1. Activar entorno
export ERICA_ENV=development
source scripts.sh

# 2. Iniciar servidor
erica-start

# 3. En otra terminal, probar
erica-health
erica-test

# 4. Ver logs
erica-logs dev
```

### Testing con Imágenes Locales
```bash
# Con CLI directo (especifica edad de la madre)
python dev_cli.py rank local test.jpg --age <EDAD_MADRE>

# Con selector interactivo
python image_selector.py

# O directo con selector
python image_selector.py --image test.jpg --age <EDAD_MADRE>

# En Docker (copiando carpeta completa)
docker cp /ruta/local/imagenes erica-api-prod:/app/temp_images/batch
docker exec erica-api-prod /bin/sh -c '
  for f in /app/temp_images/batch/*.jpg; do
    cp "$f" /app/temp_images/test.jpg;
    python dev_cli.py rank local /app/temp_images/test.jpg --age <EDAD_MADRE>;
  done
'

# Ejemplo práctico con edad 35
python dev_cli.py rank local embryo.jpg --age 35
```

### Deploy a Staging
```bash
# 1. Commit cambios
git add .
git commit -m "feat: nueva funcionalidad"

# 2. Bump version
python version_manager.py bump minor

# 3. Deploy
python deploy.py deploy staging

# 4. Verificar
python dev_cli.py health remote --env staging
```

### Deploy a Producción
```bash
# 1. Merge de staging
git checkout main
git merge staging

# 2. Release
python version_manager.py bump release
python version_manager.py tag -m "Release 2.1.0"

# 3. Deploy
python deploy.py deploy production

# 4. Verificar
curl https://erica.ivf20.app/health
```

---

## 🐳 Docker

### Build y Run
```bash
# Build
docker build -t erica-api .

# Run desarrollo
docker run -p 8001:8000 -e ERICA_ENV=development erica-api

# Run producción
docker run -p 8000:8000 -e ERICA_ENV=production erica-api
```

### Docker Compose
```bash
# Solo producción
docker-compose up -d erica-prod

# Con staging
docker-compose --profile staging up -d

# Todos
docker-compose --profile staging --profile dev up -d
```

---

## 📊 PM2 Process Manager

### Setup
```bash
# Crear configuración
python deploy.py ecosystem

# Iniciar todos
pm2 start ecosystem.config.js

# Iniciar solo producción
pm2 start ecosystem.config.js --only erica-prod
```

### Comandos
```bash
pm2 status              # Estado
pm2 logs erica-prod     # Logs
pm2 monit               # Monitor
pm2 restart erica-prod  # Reiniciar
pm2 stop erica-prod     # Detener
```

### Auto-start en boot
```bash
pm2 startup             # Configurar
pm2 save                # Guardar estado
```

---

## 🧪 Testing

### Tests de Modelos
```bash
# Suite completa
python model_tester.py test

# Benchmark
python model_tester.py benchmark --iterations 20

# Memoria
python model_tester.py memory
```

### API Testing
```bash
# Health check
curl http://localhost:8001/health

# Con API key
curl http://localhost:8001/status \
  -H "X-API-Key: dev_secret_key_12345"

# Ranking
curl -X POST http://localhost:8001/rankthisone \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev_secret_key_12345" \
  -d '{"objectId": "ABC123", "validation_key": "dev_validation_key_12345"}'
```

---

## 📝 Conda/Miniconda

### Auto-export al inicio
Al iniciar la API, si detecta un entorno conda, exporta automáticamente:
- `miniconda_requirements.yml` - Archivo YAML para recrear entorno
- `miniconda_requirements.txt` - Lista de paquetes instalados

### Manual
```bash
# Exportar environment
python dev_cli.py conda

# O directamente
python auto_requirements.py --conda
```

### Recrear entorno
```bash
# Crear nuevo entorno desde YAML
conda env create -f miniconda_requirements.yml -n erica-env

# Actualizar entorno existente
conda env update -f miniconda_requirements.yml
```

---

## 🔐 Seguridad

### API Keys
Configurar en `.env.*`:
```bash
API_SECRET_KEY=tu_clave_secreta_aqui
VALIDATION_PASS_KEY=tu_clave_validacion_aqui
```

### Endpoints protegidos
- `POST /rankthisone` - Requiere `X-API-Key`
- `GET /status` - Requiere `X-API-Key`
- `GET /debug/*` - Solo desarrollo, requiere `X-API-Key`

### Endpoints públicos
- `GET /` - Info básica
- `GET /health` - Health check

---

## 📁 Estructura de Archivos

```
api/erica-api/
├── main.py                      # Entry point FastAPI
├── config.py                    # Configuración centralizada
├── erica_api.py                # Pipeline principal
│
├── dev_cli.py                  # ⭐ CLI principal
├── auto_requirements.py        # ⭐ Gestión dependencias + conda
├── model_tester.py             # ⭐ Testing modelos
├── image_selector.py           # ⭐ Selector imágenes
├── deploy.py                   # ⭐ Deployment manager
├── version_manager.py          # ⭐ Versionado
├── scripts.sh                  # ⭐ Scripts bash
│
├── .env.development            # Config desarrollo
├── .env.staging                # Config staging
├── .env.production             # Config producción
├── requirements.txt            # Dependencias pip
├── miniconda_requirements.yml  # Entorno conda (auto-generado)
├── Dockerfile                  # Docker config
├── docker-compose.yml          # Docker Compose
├── ecosystem.config.js         # PM2 config
│
├── COMANDOS.md                 # 📖 Referencia comandos
├── DEV_TOOLS.md               # 📖 Esta guía
│
├── models/                     # Modelos ML
├── utils/                      # Utilidades
└── logs/                       # Logs
```

---

## 🆘 Troubleshooting

### Requirements desactualizados
```bash
python auto_requirements.py
pip install -r requirements.txt
```

### Modelos no cargan
```bash
python model_tester.py files
python model_tester.py test
```

### Puerto ocupado
```bash
# Ver qué está usando el puerto
lsof -i :8001

# Matar proceso
lsof -ti :8001 | xargs kill -9
```

### PM2 no responde
```bash
pm2 kill
pm2 resurrect
```

### Conda environment corrupto
```bash
# Recrear desde archivo
conda env remove -n erica-env
conda env create -f miniconda_requirements.yml -n erica-env
```

---

## 📚 Recursos

- **COMANDOS.md** - Referencia completa de comandos
- **README.md** - Documentación general del proyecto
- **Logs**: `./logs/`
- **Health check producción**: https://erica.ivf20.app/health
- **API docs** (dev): http://localhost:8001/docs

---

## 💡 Tips

1. **Usa `source scripts.sh`** para tener todos los comandos disponibles en tu terminal
2. **Verifica el entorno** antes de hacer deploy: `python dev_cli.py env validate`
3. **Prueba localmente** antes de subir: `python image_selector.py`
4. **Monitorea con PM2**: `pm2 monit` para ver CPU/memoria en tiempo real
5. **Auto-export conda**: El entorno se exporta automáticamente al iniciar la API
6. **Bump semántico**: patch para fixes, minor para features, major para breaking changes
