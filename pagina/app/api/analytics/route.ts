import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Generar datos analíticos simulados
    const hourlyData = generateHourlyData()
    const stationData = generateStationData()
    const weeklyData = generateWeeklyData()
    const metrics = generateMetrics()

    return NextResponse.json({
      success: true,
      data: {
        hourly: hourlyData,
        stations: stationData,
        weekly: weeklyData,
        metrics: metrics
      },
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    console.error('Error obteniendo analytics:', error)
    return NextResponse.json(
      { error: 'Error interno del servidor' },
      { status: 500 }
    )
  }
}

function generateHourlyData() {
  const hours = Array.from({length: 24}, (_, i) => i)
  return hours.map(hour => {
    let baseTrips = 20
    
    // Patrones de uso típicos
    if (hour >= 6 && hour <= 9) baseTrips = Math.random() * 100 + 80 // Pico matutino
    else if (hour >= 17 && hour <= 20) baseTrips = Math.random() * 120 + 90 // Pico vespertino
    else if (hour >= 11 && hour <= 14) baseTrips = Math.random() * 60 + 40 // Almuerzo
    else baseTrips = Math.random() * 30 + 10 // Horas bajas

    return {
      hour: `${hour.toString().padStart(2, '0')}:00`,
      trips: Math.round(baseTrips),
      predictions: Math.round(baseTrips * (0.9 + Math.random() * 0.2))
    }
  })
}

function generateStationData() {
  const stations = [
    'Plaza de Mayo', 'Puerto Madero', 'Recoleta', 'Palermo', 'San Telmo',
    'Belgrano', 'Villa Crick', 'Barracas', 'La Boca', 'Constitución'
  ]
  
  return stations.map(station => ({
    station,
    trips: Math.round(Math.random() * 500 + 100),
    availability: Math.round(Math.random() * 100),
    rating: Math.round((Math.random() * 2 + 3) * 10) / 10 // 3.0 - 5.0
  }))
}

function generateWeeklyData() {
  const days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
  const baseValues = [850, 920, 880, 900, 750, 450, 380]
  
  return days.map((day, index) => ({
    day,
    trips: Math.round(baseValues[index] + (Math.random() * 100 - 50))
  }))
}

function generateMetrics() {
  return {
    totalStations: 402,
    activeBikes: Math.round(Math.random() * 500 + 3500),
    totalTrips: Math.round(Math.random() * 2000 + 14000),
    avgWaitTime: Math.round((Math.random() * 2 + 2) * 10) / 10, // 2.0 - 4.0 min
    systemUptime: 99.2,
 