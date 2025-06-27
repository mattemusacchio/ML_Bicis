import { NextResponse } from 'next/server'
import path from 'path'
import { promises as fs } from 'fs'

export async function GET() {
  try {
    // Leer datos de estaciones desde el archivo JSON
    const jsonDirectory = path.join(process.cwd(), 'public')
    const fileContents = await fs.readFile(jsonDirectory + '/stations_data.json', 'utf8')
    const data = JSON.parse(fileContents)
    
    return NextResponse.json({
      success: true,
      data: data.stations,
      metadata: data.metadata
    })
  } catch (error) {
    console.error('Error loading stations data:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to load stations data',
        data: []
      }, 
      { status: 500 }
    )
  }
}

// Opcional: endpoint para obtener datos en tiempo real (simulado)
export async function POST() {
  try {
    // Simular actualización de datos en tiempo real
    const jsonDirectory = path.join(process.cwd(), 'public')
    const fileContents = await fs.readFile(jsonDirectory + '/stations_data.json', 'utf8')
    const data = JSON.parse(fileContents)
    
    // Simular cambios en disponibilidad
    const updatedStations = data.stations.map(station => ({
      ...station,
      available_bikes: Math.max(0, station.available_bikes + Math.floor(Math.random() * 6 - 3)),
      last_updated: new Date().toISOString()
    }))
    
    return NextResponse.json({
      success: true,
      data: updatedStations,
      metadata: {
        ...data.metadata,
        last_updated: new Date().toISOString(),
        note: 'Real-time simulation'
      }
    })
  } catch (error) {
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to update stations data'
      }, 
      { status: 500 }
    )
  }
}