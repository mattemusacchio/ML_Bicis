# 🚴 BA Bicis Dashboard

Dashboard interactivo para visualización y predicciones de BA Bicis en tiempo real.

## 🌟 Características

- **📊 Visualización de datos**: Gráficos interactivos con Plotly
- **🔮 Predicciones ML**: Predicciones de arribos en tiempo real
- **🗺️ Mapa interactivo**: Ubicación y estado de estaciones
- **📱 Responsive**: Diseño adaptativo para móviles y desktop
- **⚡ Tiempo real**: Actualización automática de datos

## 🚀 Inicio Rápido

### Instalación

```bash
# Navegar a la carpeta del proyecto
cd pagina

# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev
```

La aplicación estará disponible en `http://localhost:3000`

### Desarrollo

```bash
# Modo desarrollo
npm run dev

# Build para producción
npm run build

# Iniciar servidor de producción
npm start

# Linting
npm run lint
```

## 🔧 Configuración para Vercel

### Despliegue automático

1. **Conectar repositorio a Vercel**:
   - Ve a [vercel.com](https://vercel.com)
   - Importa tu repositorio
   - Configura la carpeta raíz como `pagina`

2. **Configuración del proyecto**:
   - Framework: Next.js
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`

### Variables de entorno (opcional)

Crea un archivo `.env.local` para variables de entorno:

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:3000/api

# Model Configuration
MODEL_PATH=/path/to/your/model
```

## 📁 Estructura del Proyecto

```
pagina/
├── app/                    # App Router de Next.js
│   ├── api/               # API Routes
│   │   ├── analytics/     # Endpoints de análisis
│   │   ├── predictions/   # Endpoints de predicciones
│   │   └── stations/      # Endpoints de estaciones
│   ├── globals.css        # Estilos globales
│   ├── layout.tsx         # Layout principal
│   └── page.tsx          # Página principal
├── components/            # Componentes React
│   ├── DataVisualization.tsx
│   ├── PredictionInterface.tsx
│   ├── RealTimeMetrics.tsx
│   └── StationMap.tsx
├── package.json
├── next.config.js
├── tailwind.config.js
└── README.md
```

## 🎯 Funcionalidades

### 1. Resumen General
- Métricas en tiempo real
- Gráficos de uso por hora
- Análisis de tendencias

### 2. Predicciones
- Selección de estación y horario
- Predicciones con nivel de confianza
- Historial de predicciones

### 3. Mapa Interactivo
- Ubicación de todas las estaciones
- Estado en tiempo real
- Filtros por estado
- Información detallada por estación

### 4. Análisis Avanzado
- Patrones semanales
- Distribución por estaciones
- Métricas de rendimiento

## 🔌 Integración con Modelo ML

Para conectar con tu modelo real, modifica los archivos en `app/api/`:

```typescript
// app/api/predictions/route.ts
import { loadModel } from '@/lib/model'

export async function POST(request: NextRequest) {
  const model = await loadModel()
  const prediction = await model.predict(features)
  // ...
}
```

## 🎨 Personalización

### Colores y Tema

Modifica `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      'ba-blue': '#tu-color-azul',
      'ba-green': '#tu-color-verde',
    }
  }
}
```

### Configuración de Gráficos

Los gráficos usan Plotly.js. Personaliza en `components/DataVisualization.tsx`:

```typescript
const layout = {
  title: 'Tu Título',
  colorway: ['#color1', '#color2'],
  // más configuraciones...
}
```

## 📊 Datos

### Estructura de Datos

```typescript
interface Station {
  id: string
  name: string
  lat: number
  lng: number
  availableBikes: number
  totalDocks: number
  status: 'active' | 'maintenance' | 'full'
}

interface Prediction {
  stationId: string
  predicted: number
  confidence: number
  timestamp: string
}
```

### Fuentes de Datos

- **Estaciones**: API simulada con datos de CABA
- **Predicciones**: Modelo ML simulado
- **Analytics**: Datos históricos simulados
- **Tiempo real**: Actualización cada 30 segundos

## 🚀 Despliegue en Vercel

### Opción 1: Desde la interfaz web

1. Ve a [vercel.com](https://vercel.com)
2. Conecta tu repositorio de GitHub
3. Configura la carpeta raíz como `pagina`
4. Deploy automático

### Opción 2: Desde CLI

```bash
# Instalar Vercel CLI
npm i -g vercel

# Navegar a carpeta pagina
cd pagina

# Desplegar
vercel

# Configurar dominio personalizado (opcional)
vercel --prod
```

## 🔧 Troubleshooting

### Problemas comunes

1. **Error de módulos**: Ejecuta `npm install` en la carpeta `pagina`
2. **Mapa no carga**: Verifica que Leaflet esté instalado
3. **Gráficos no aparecen**: Plotly.js necesita renderizado del lado cliente
4. **Build falla**: Revisa las dependencias en `package.json`

### Logs de desarrollo

```bash
# Ver logs detallados
npm run dev -- --verbose

# Ver logs de build
npm run build -- --debug
```

## 📈 Métricas y Monitoreo

Una vez desplegado en Vercel, puedes monitorear:

- **Analytics**: Visitas, tiempo de carga
- **Funciones**: Uso de API routes
- **Performance**: Core Web Vitals
- **Errores**: Logs de aplicación

## 🤝 Contribuir

1. Fork del repositorio
2. Crear feature branch
3. Commit cambios
4. Push al branch
5. Crear Pull Request

## 📝 Licencia

Este proyecto es parte de un trabajo académico de Machine Learning.

## 📞 Soporte

Si tienes problemas:

1. Revisa este README
2. Consulta la documentación de Next.js
3. Verifica las issues en GitHub
4. Contacta al desarrollador

---

¡Disfruta usando el dashboard de BA Bicis! 🚴‍♀️🚴‍♂️ 