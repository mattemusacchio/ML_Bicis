import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET() {
  try {
    // Leer el archivo CSV de datos reales usando streaming para archivos grandes
    const csvPath = path.join(process.cwd(), '../data/processed/trips.csv')
    
    // Verificar si el archivo existe
    if (!fs.existsSync(csvPath)) {
      // Si no existe, devolver datos simulados
      return NextResponse.json({
        success: true,
        data: generateMockHeatmapData(),
        source: 'simulated'
      })
    }

    // Leer solo las primeras líneas del archivo para evitar problemas de memoria
    const lines = await readFirstLines(csvPath, 5000) // Solo primeras 5000 líneas
    
    // Procesar datos para el heatmap
    const heatmapData = []
    const stationStats = new Map()
    
    // Procesar los registros obtenidos (ya limitados por readFirstLines)
    const maxRecords = lines.length
    
    for (let i = 0; i < maxRecords; i++) {
      const line = lines[i].trim()
      if (!line) continue
      
      const columns = parseCSVLine(line)
      if (columns.length < 12) continue
      
      const latOrigen = parseFloat(columns[6])
      const longOrigen = parseFloat(columns[5])
      const latDestino = parseFloat(columns[11])
      const longDestino = parseFloat(columns[10])
      const nombreOrigen = columns[3]
      const nombreDestino = columns[8]
      const duracion = parseInt(columns[1])
      
      // Validar coordenadas
      if (isValidCoordinate(latOrigen, longOrigen)) {
        heatmapData.push({
          lat: latOrigen,
          lng: longOrigen,
          intensity: 1,
          type: 'origen',
          station: nombreOrigen,
          duration: duracion
        })
        
        // Estadísticas por estación
        const key = `${nombreOrigen}_origen`
        stationStats.set(key, (stationStats.get(key) || 0) + 1)
      }
      
      if (isValidCoordinate(latDestino, longDestino)) {
        heatmapData.push({
          lat: latDestino,
          lng: longDestino,
          intensity: 1,
          type: 'destino',
          station: nombreDestino,
          duration: duracion
        })
        
        const key = `${nombreDestino}_destino`
        stationStats.set(key, (stationStats.get(key) || 0) + 1)
      }
    }

    // Generar estadísticas de estaciones
    const topStations = Array.from(stationStats.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([station, count]) => ({
        station: station.replace(/_origen|_destino/, ''),
        count,
        type: station.includes('_origen') ? 'origen' : 'destino'
      }))

    return NextResponse.json({
      success: true,
      data: heatmapData,
      stats: {
        totalTrips: heatmapData.length / 2,
        topStations,
        processedRecords: maxRecords
      },
      source: 'real'
    })
    
  } catch (error) {
    console.error('Error procesando datos:', error)
    return NextResponse.json({
      success: true,
      data: generateMockHeatmapData(),
      source: 'fallback'
    })
  }
}

async function readFirstLines(filePath: string, maxLines: number): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const lines: string[] = []
    const readable = fs.createReadStream(filePath, { encoding: 'utf-8' })
    let remainder = ''
    let lineCount = 0
    
    readable.on('data', (chunk: string) => {
      const data = remainder + chunk
      const lineArray = data.split('\n')
      remainder = lineArray.pop() || ''
      
      for (const line of lineArray) {
        if (lineCount === 0) {
          // Saltar header
          lineCount++
          continue
        }
        
        if (lineCount >= maxLines + 1) {
          readable.destroy()
          resolve(lines)
          return
        }
        
        lines.push(line.trim())
        lineCount++
      }
    })
    
    readable.on('end', () => {
      if (remainder) {
        lines.push(remainder.trim())
      }
      resolve(lines)
    })
    
    readable.on('error', reject)
  })
}

function parseCSVLine(line: string): string[] {
  const result = []
  let current = ''
  let inQuotes = false
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    
    if (char === '"') {
      inQuotes = !inQuotes
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim())
      current = ''
    } else {
      current += char
    }
  }
  
  result.push(current.trim())
  return result
}

function isValidCoordinate(lat: number, lng: number): boolean {
  return !isNaN(lat) && !isNaN(lng) && 
         lat >= -35 && lat <= -34 &&  // Buenos Aires lat range
         lng >= -59 && lng <= -58     // Buenos Aires lng range
}

function generateMockHeatmapData() {
  const mockData = []
  const stations = [
    { name: 'Plaza de Mayo', lat: -34.6081, lng: -58.3698, intensity: 50 },
    { name: 'Puerto Madero', lat: -34.6118, lng: -58.3623, intensity: 45 },
    { name: 'Recoleta', lat: -34.5875, lng: -58.3974, intensity: 40 },
    { name: 'Palermo', lat: -34.5755, lng: -58.4338, intensity: 60 },
    { name: 'San Telmo', lat: -34.6214, lng: -58.3731, intensity: 35 },
    { name: 'Belgrano', lat: -34.5633, lng: -58.4606, intensity: 30 },
    { name: 'Villa Crick', lat: -34.6037, lng: -58.4427, intensity: 25 },
    { name: 'Barracas', lat: -34.6489, lng: -58.3759, intensity: 20 }
  ]

  stations.forEach(station => {
    // Agregar puntos aleatorios alrededor de cada estación
    for (let i = 0; i < station.intensity; i++) {
      mockData.push({
        lat: station.lat + (Math.random() - 0.5) * 0.01,
        lng: station.lng + (Math.random() - 0.5) * 0.01,
        intensity: Math.random() * 3 + 1,
        type: Math.random() > 0.5 ? 'origen' : 'destino',
        station: station.name,
        duration: Math.round(Math.random() * 1800 + 300)
      })
    }
  })

  return mockData
} 