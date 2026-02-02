# ERICA API - Guía de Seguridad

## 🔐 Sistema de Autenticación

La API ERICA utiliza un sistema de **doble autenticación** para el endpoint de ranking:

1. **API Key** (Header) - Autenticación general de la API
2. **Validation Key** (Body) - Validación específica de operación de ranking

---

## 🔑 Llaves de Seguridad

### 1. API Secret Key
- **Variable de entorno**: `API_SECRET_KEY`
- **Ubicación**: Header HTTP `X-API-Key`
- **Uso**: Autenticar todas las llamadas a endpoints protegidos
- **Scope**: API completa

### 2. Validation Pass Key
- **Variable de entorno**: `VALIDATION_PASS_KEY`
- **Ubicación**: Body del request JSON como `validation_key`
- **Uso**: Validar específicamente operaciones de ranking
- **Scope**: Solo endpoint `/rankthisone`

---

## ⚙️ Configuración

### Archivo `.env.development`
```bash
# Desarrollo - Llaves de ejemplo (NO usar en producción)
API_SECRET_KEY=dev_secret_key_12345
VALIDATION_PASS_KEY=dev_validation_key_12345
```

### Archivo `.env.staging`
```bash
# Staging - Llaves de pre-producción
API_SECRET_KEY=staging_secret_key_CHANGE_ME
VALIDATION_PASS_KEY=staging_validation_key_CHANGE_ME
```

### Archivo `.env.production`
```bash
# Producción - GENERAR LLAVES SEGURAS
API_SECRET_KEY=GENERAR_CLAVE_SEGURA_AQUI
VALIDATION_PASS_KEY=GENERAR_OTRA_CLAVE_SEGURA_AQUI
```

**⚠️ IMPORTANTE**: 
- Nunca commitear las llaves de staging/production al repositorio
- Usar llaves largas y aleatorias en producción
- Rotar llaves periódicamente
- Los archivos `.env.*` están en `.gitignore` por seguridad

### Generar Llaves Seguras
```bash
# En Linux/Mac
openssl rand -hex 32

# O en Python
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 📡 Endpoints y Seguridad

### Públicos (sin autenticación)
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información básica de la API |
| `/health` | GET | Health check |

**Ejemplo:**
```bash
curl https://erica.ivf20.app/health
```

---

### Protegidos (requieren X-API-Key)
| Endpoint | Método | Headers Requeridos | Body |
|----------|--------|-------------------|------|
| `/status` | GET | `X-API-Key` | - |
| `/debug/config` | GET | `X-API-Key` | - |
| `/debug/test-pipeline` | POST | `X-API-Key` | JSON |

**Ejemplo:**
```bash
curl https://erica.ivf20.app/status \
  -H "X-API-Key: TU_API_SECRET_KEY"
```

---

### Ranking (requiere ambas llaves)
| Endpoint | Método | Headers | Body |
|----------|--------|---------|------|
| `/rankthisone` | POST | `X-API-Key`, `Content-Type` | `objectId`, `validation_key` |

**Ejemplo:**
```bash
curl -X POST https://erica.ivf20.app/rankthisone \
  -H "Content-Type: application/json" \
  -H "X-API-Key: TU_API_SECRET_KEY" \
  -d '{
    "objectId": "CYCLE_OBJECT_ID",
    "validation_key": "TU_VALIDATION_PASS_KEY"
  }'
```

---

## 🔄 Migración de Llamadas Antiguas

### Si tu código actual NO usa autenticación:

#### ❌ ANTES (sin seguridad)
```javascript
fetch('https://erica.ivf20.app/rankthisone', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ objectId: 'ABC123' })
})
```

#### ✅ DESPUÉS (con seguridad)
```javascript
fetch('https://erica.ivf20.app/rankthisone', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.API_SECRET_KEY  // ⬅️ AGREGAR
  },
  body: JSON.stringify({
    objectId: 'ABC123',
    validation_key: process.env.VALIDATION_PASS_KEY  // ⬅️ AGREGAR
  })
})
```

---

### Desde Parse Cloud Functions

#### ❌ ANTES
```javascript
Parse.Cloud.define("rankCycle", async (request) => {
  const response = await Parse.Cloud.httpRequest({
    url: 'https://erica.ivf20.app/rankthisone',
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: { objectId: request.params.cycleId }
  });
  return response.data;
});
```

#### ✅ DESPUÉS
```javascript
Parse.Cloud.define("rankCycle", async (request) => {
  const response = await Parse.Cloud.httpRequest({
    url: 'https://erica.ivf20.app/rankthisone',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': process.env.API_SECRET_KEY  // ⬅️ AGREGAR
    },
    body: {
      objectId: request.params.cycleId,
      validation_key: process.env.VALIDATION_PASS_KEY  // ⬅️ AGREGAR
    }
  });
  return response.data;
});
```

---

### Desde Python/Requests

#### ❌ ANTES
```python
import requests

response = requests.post(
    'https://erica.ivf20.app/rankthisone',
    json={'objectId': 'ABC123'}
)
```

#### ✅ DESPUÉS
```python
import requests
import os

response = requests.post(
    'https://erica.ivf20.app/rankthisone',
    headers={
        'Content-Type': 'application/json',
        'X-API-Key': os.getenv('API_SECRET_KEY')  # ⬅️ AGREGAR
    },
    json={
        'objectId': 'ABC123',
        'validation_key': os.getenv('VALIDATION_PASS_KEY')  # ⬅️ AGREGAR
    }
)
```

---

## 🚨 Manejo de Errores

### 403 Forbidden - API Key inválida o faltante
```json
{
  "detail": "Invalid or missing API key"
}
```

**Solución**: Verificar que el header `X-API-Key` esté presente y tenga el valor correcto.

---

### 403 Forbidden - Validation Key inválida
```json
{
  "status": 403,
  "error": "Invalid validation key."
}
```

**Solución**: Verificar que `validation_key` en el body coincida con `VALIDATION_PASS_KEY`.

---

### 400 Bad Request - Falta objectId o validation_key
```json
{
  "status": 400,
  "error": "Missing 'objectId' or 'validation_key'."
}
```

**Solución**: Asegurarse de enviar ambos campos en el body del request.

---

## 🧪 Testing

### Verificar Configuración Local
```bash
# 1. Ver configuración actual
python dev_cli.py env show

# 2. Verificar llaves
python dev_cli.py env validate
```

### Test de Health Check (público)
```bash
curl -v http://localhost:8001/health
# Debe retornar 200 OK sin autenticación
```

### Test de Status (requiere API Key)
```bash
# ❌ Sin API Key - debe fallar con 403
curl -v http://localhost:8001/status

# ✅ Con API Key - debe funcionar
curl -v http://localhost:8001/status \
  -H "X-API-Key: dev_secret_key_12345"
```

### Test de Ranking (requiere ambas llaves)
```bash
# Con las llaves de desarrollo
curl -X POST http://localhost:8001/rankthisone \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev_secret_key_12345" \
  -d '{
    "objectId": "TEST_CYCLE_ID",
    "validation_key": "dev_validation_key_12345"
  }'
```

---

## 🔒 Mejores Prácticas

### ✅ DO - Hacer
1. Almacenar llaves en variables de entorno
2. Usar llaves diferentes para cada entorno
3. Generar llaves largas y aleatorias para producción
4. Rotar llaves periódicamente
5. Nunca hardcodear llaves en el código
6. Usar HTTPS en producción

### ❌ DON'T - No Hacer
1. Commitear archivos `.env.*` con llaves reales
2. Compartir llaves por email o chat
3. Usar las mismas llaves en dev y producción
4. Exponer llaves en logs
5. Usar llaves débiles o predecibles

---

## 📋 Checklist de Migración

- [ ] Obtener `API_SECRET_KEY` del administrador o generarla
- [ ] Obtener `VALIDATION_PASS_KEY` del administrador o generarla
- [ ] Configurar variables en `.env.*` o en tu sistema de secrets
- [ ] Actualizar todas las llamadas a `/rankthisone`:
  - [ ] Agregar header `X-API-Key`
  - [ ] Agregar campo `validation_key` en body
- [ ] Probar en desarrollo
- [ ] Probar en staging
- [ ] Desplegar a producción
- [ ] Verificar logs para confirmar funcionamiento

---

## 🆘 Soporte

### ¿Dónde encuentro las llaves?
1. **Desarrollo**: En `.env.development` (valores por defecto incluidos)
2. **Staging/Producción**: Solicitar al administrador del sistema

### ¿Cómo verifico que mis llaves funcionan?
```bash
# Test rápido con dev_cli
python dev_cli.py health local

# Test manual
curl http://localhost:8001/status \
  -H "X-API-Key: TU_API_SECRET_KEY"
```

### ¿Qué hago si olvidé mi API Key?
Contactar al administrador para generar nuevas llaves o revisar el archivo `.env.*` del servidor.

---

## 📚 Documentación Relacionada

- [COMANDOS.md](./COMANDOS.md) - Referencia completa de comandos
- [DEV_TOOLS.md](./DEV_TOOLS.md) - Herramientas de desarrollo
- [README.md](./README.md) - Documentación general
