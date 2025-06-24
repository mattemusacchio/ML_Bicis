import { NextResponse } from 'next/server'

const MOCK_STATIONS = [
  {
    id: '1',
    name: 'Plaza de Mayo',
    lat: -34.6081,
    lng: -58.3698,
    availableBikes: 12,
    totalDocks: 20,
    status: 'active',
    lastUpdate: new Date().toISOString()
  },
  {
    id: '2',
    name: 'Puerto Madero',
    lat: -34.6118,
    lng: -58.3623,
    availableBikes: 8,
    totalDocks: 15,
    status: 'active',
    lastUpdate: new Date().toISOString()
  },
  {
    id: '3',
    name: 'Recoleta',
    lat: -34.5875,
    lng: -58.3974,
    availableBikes: 0,
    totalDocks: 25,
    status: 'full',
    lastUpdate: new Date().toISOString()
  },
  {
    id: '4',
    name: 'Palermo',
    lat: -34.5755,
    lng: -58.4338,
    availableBikes: 18,
    totalDocks: 30,
    status: 'active',
    lastUpdate: new Date().toISOString()
  },
  {
    id: '5',
    name: 'San Telmo',
    lat: -34.6214,
    lng: -58.3731,
    availableBikes: 5,
    totalDocks: 20,
    status: 'maintenance',
    lastUpdate: new Date().toISOString()
  }
]

export async function GET() {
  try {
    // Simular variación en datos en tiempo real
    const stationsWithVariation = MOCK_STATIONS.map(station => ({
      ...station,
      availableBikes: Math.max(0, station.availableBikes + Math.floor(Math.random() * 6 - 3)),
      lastUpdate: new Date().toISOString()
    }))

    return NextResponse.json({
      success: true,
      data: stationsWithVariation,
      totalStations: stationsWithVariation.length,
      lastUpdate: new Date().toISOString()
    })
  } catch (error) {
    console.error('Error obteniendo estaciones:', error)
    return NextResponse.json(
      { error: 'Error interno del servidor' },
      { status: 500 }
    )
  }
} 