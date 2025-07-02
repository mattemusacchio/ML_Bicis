import { NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

interface TripHeatmapRequest {
  startDate?: string;
  endDate?: string;
  timeRange?: 'hourly' | 'daily' | 'weekly';
  hour?: number;
  dayOfWeek?: number;
}

export async function POST(request: Request) {
  try {
    const params: TripHeatmapRequest = await request.json()
    
    // Llamar script de Python para procesar datos reales
    const pythonScript = path.join(process.cwd(), '..', 'scripts', 'process_trips_heatmap.py')
    
    const result = await new Promise((resolve, reject) => {
      const python = spawn('python', [pythonScript, JSON.stringify(params)])
      
      let output = ''
      let error = ''
      
      python.stdout.on('data', (data) => {
        output += data.toString()
      })
      
      python.stderr.on('data', (data) => {
        error += data.toString()
      })
      
      python.on('close', (code) => {
        if (code === 0) {
          try {
            resolve(JSON.parse(output))
          } catch (e) {
            reject(new Error('Invalid JSON output from Python script'))
          }
        } else {
          reject(new Error(`Python script failed: ${error}`))
        }
      })
    })
    
    return NextResponse.json(result)
    
  } catch (error) {
    console.error('Error processing trips heatmap:', error)
    
    // Fallback con datos simulados más realistas
    const fallbackData = generateRealisticFallback()
    
    return NextResponse.json({
      success: true,
      data: fallbackData.trips,
      summary: fallbackData.summary,
      source: 'fallback',
      message: 'Usando datos simulados - archivo real no disponible'
    })
  }
}

function generateRealisticFallback() {
  // Datos de estaciones reales de Buenos Aires (subset)
  const stations = [
    { id: 1, name: "Retiro", lat: -34.5925, lng: -58.3747 },
    { id: 2, name: "Puerto Madero", lat: -34.6118, lng: -58.3623 },
    { id: 3, name: "San Telmo", lat: -34.6158, lng: -58.3731 },
    { id: 4, name: "Recoleta", lat: -34.5875, lng: -58.3974 },
    { id: 5, name: "Palermo", lat: -34.5755, lng: -58.4338 },
    { id: 6, name: "Centro", lat: -34.6037, lng: -58.3816 },
    { id: 7, name: "Belgrano", lat: -34.5633, lng: -58.4606 },
    { id: 8, name: "Villa Crick", lat: -34.6489, lng: -58.3759 }
  ]
  
  const trips = []
  let totalTrips = 0
  
  // Generar patrones de viajes más realistas por hora
  for (let hour = 0; hour < 24; hour++) {
    for (let i = 0; i < stations.length; i++) {
      for (let j = 0; j < stations.length; j++) {
        if (i !== j) {
          // Patrones de demanda más realistas
          let intensity = 0.1 // base
          
          // Picos de mañana (7-9 AM)
          if (hour >= 7 && hour <= 9) {
            intensity += 0.6
          }
          // Picos de tarde (17-19 PM)
          if (hour >= 17 && hour <= 19) {
            intensity += 0.5
          }
          // Horario laboral
          if (hour >= 9 && hour <= 17) {
            intensity += 0.2
          }
          // Fin de semana (simulado para sábado)
          if (hour >= 10 && hour <= 22) {
            intensity += 0.3
          }
          
          const tripCount = Math.round(intensity * 50 * Math.random())
          totalTrips += tripCount
          
          if (tripCount > 0) {
            trips.push({
              origen_id: stations[i].id,
              origen_name: stations[i].name,
              origen_lat: stations[i].lat,
              origen_lng: stations[i].lng,
              destino_id: stations[j].id,
              destino_name: stations[j].name,
              destino_lat: stations[j].lat,
              destino_lng: stations[j].lng,
              trip_count: tripCount,
              hour: hour,
              avg_duration: Math.round(300 + Math.random() * 1200), // 5-25 minutos
              intensity: Math.min(intensity, 1.0)
            })
          }
        }
      }
    }
  }
  
  const summary = {
    totalTrips,
    totalStations: stations.length,
    dateRange: '2024-01-01 to 2024-12-31',
    peakHours: [8, 18],
    avgTripDuration: 900 // 15 minutos
  }
  
  return { trips, summary }
}

export async function GET(request: Request) {
  try {
    // Extraer parámetros de la URL
    const { searchParams } = new URL(request.url);
    const hour = searchParams.get('hour') || '8';
    const date = searchParams.get('date') || '2024-03-15';
    const params = {
      hour: parseInt(hour),
      date: date
    };
    
    console.log('GET request params:', params);
    
    // Llamar script de Python para procesar datos reales
    const pythonScript = path.join(process.cwd(), '..', 'scripts', 'process_trips_heatmap.py')
    
    const result = await new Promise((resolve, reject) => {
      const python = spawn('python', [pythonScript, JSON.stringify(params)])
      
      let output = ''
      let error = ''
      
      python.stdout.on('data', (data) => {
        output += data.toString()
      })
      
      python.stderr.on('data', (data) => {
        error += data.toString()
      })
      
      python.on('close', (code) => {
        if (code === 0) {
          try {
            const parsedResult = JSON.parse(output);
            console.log('Python script returned:', parsedResult.data ? parsedResult.data.length + ' stations' : 'no data');
            resolve(parsedResult);
          } catch (e) {
            console.error('JSON parse error:', e);
            reject(new Error('Invalid JSON output from Python script'))
          }
        } else {
          console.error('Python script error:', error);
          reject(new Error(`Python script failed: ${error}`))
        }
      })
    })
    
    return NextResponse.json(result)
    
  } catch (error) {
    console.error('Error processing trips heatmap:', error)
    
    // Fallback con datos simulados más realistas
    const fallbackData = generateRealisticFallback()
    
    return NextResponse.json({
      success: true,
      data: fallbackData.trips,
      summary: fallbackData.summary,
      source: 'fallback',
      message: 'Usando datos simulados - archivo real no disponible'
    })
  }
} 