'use client'

import React, { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'

// Importar el mapa dinámicamente para evitar errores de SSR
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
)
const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
)
const Marker = dynamic(
  () => import('react-leaflet').then((mod) => mod.Marker),
  { ssr: false }
)
const Popup = dynamic(
  () => import('react-leaflet').then((mod) => mod.Popup),
  { ssr: false }
)

interface Station {
  id: string;
  name: string;
  lat: number;
  lng: number;
  availableBikes: number;
  totalDocks: number;
  status: 'active' | 'maintenance' | 'full' | 'empty';
  address?: string;
  totalTrips?: number;
  popularityScore?: number;
}

export default function StationMap() {
  const [stations, setStations] = useState<Station[]>([])
  const [selectedFilter, setSelectedFilter] = useState('all')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Cargar datos reales de estaciones desde la API
    const fetchRealStations = async () => {
      try {
        setIsLoading(true)
        
        // Intentar cargar desde la API primero
        const response = await fetch('/api/stations')
        
        if (response.ok) {
          const result = await response.json()
          if (result.success && result.data) {
            const realStations: Station[] = result.data.map((station: any) => ({
              id: station.id,
              name: station.name,
              lat: station.lat,
              lng: station.lng,
              availableBikes: station.available_bikes,
              totalDocks: station.total_docks,
              status: station.status,
              address: station.address,
              totalTrips: station.total_trips,
              popularityScore: station.popularity_score
            }))
            
            setStations(realStations)
            console.log(`✅ Cargadas ${realStations.length} estaciones reales desde API`)
            return
          }
        }
        
        // Fallback: cargar desde archivo JSON estático
        const jsonResponse = await fetch('/stations_data.json')
        if (jsonResponse.ok) {
          const jsonData = await jsonResponse.json()
          const realStations: Station[] = jsonData.stations.map((station: any) => ({
            id: station.id,
            name: station.name,
            lat: station.lat,
            lng: station.lng,
            availableBikes: station.available_bikes,
            totalDocks: station.total_docks,
            status: station.status,
            address: station.address,
            totalTrips: station.total_trips,
            popularityScore: station.popularity_score
          }))
          
          setStations(realStations)
          console.log(`✅ Cargadas ${realStations.length} estaciones reales desde JSON`)
          return
        }
        
        throw new Error('No se pudieron cargar los datos de estaciones')
        
      } catch (error) {
        console.error('❌ Error cargando estaciones reales:', error)
        
        // Fallback a datos de muestra en caso de error
        const fallbackStations: Station[] = [
          {
            id: '1',
            name: 'Estación de prueba',
            lat: -34.6081,
            lng: -58.3698,
            availableBikes: 10,
            totalDocks: 20,
            status: 'active'
          }
        ]
        setStations(fallbackStations)
        console.log('⚠️ Usando datos de fallback')
      } finally {
        setIsLoading(false)
      }
    }

    fetchRealStations()
  }, [])

  const filteredStations = stations.filter(station => {
    if (selectedFilter === 'all') return true
    return station.status === selectedFilter
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return '#28a745'
      case 'maintenance': return '#ffc107'
      case 'full': return '#dc3545'
      default: return '#6c757d'
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active': return 'Activa'
      case 'maintenance': return 'Mantenimiento'
      case 'full': return 'Completa'
      default: return 'Desconocido'
    }
  }

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="h-96 bg-gray-200 rounded"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Filtros */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">🔍 Filtrar Estaciones</h3>
        <div className="flex flex-wrap gap-2">
          {[
            { key: 'all', label: 'Todas', count: stations.length },
            { key: 'active', label: 'Activas', count: stations.filter(s => s.status === 'active').length },
            { key: 'maintenance', label: 'Mantenimiento', count: stations.filter(s => s.status === 'maintenance').length },
            { key: 'full', label: 'Completas', count: stations.filter(s => s.status === 'full').length }
          ].map(filter => (
            <button
              key={filter.key}
              onClick={() => setSelectedFilter(filter.key)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedFilter === filter.key
                  ? 'bg-ba-blue text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {filter.label} ({filter.count})
            </button>
          ))}
        </div>
      </div>

      {/* Mapa */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">🗺️ Ubicaciones de Estaciones</h3>
        <div className="h-96 rounded-lg overflow-hidden">
          {typeof window !== 'undefined' && (
            <MapContainer
              center={[-34.6037, -58.3816]}
              zoom={12}
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              {filteredStations.map(station => (
                <Marker
                  key={station.id}
                  position={[station.lat, station.lng]}
                >
                  <Popup>
                    <div className="p-2 min-w-48">
                      <h4 className="font-semibold text-gray-900 mb-2">{station.name}</h4>
                      <div className="space-y-1 text-sm">
                        <p>
                          <span className="font-medium">Estado:</span>
                          <span 
                            className="ml-2 px-2 py-1 rounded-full text-xs text-white"
                            style={{ backgroundColor: getStatusColor(station.status) }}
                          >
                            {getStatusText(station.status)}
                          </span>
                        </p>
                        <p>
                          <span className="font-medium">Bicis disponibles:</span> {station.availableBikes}
                        </p>
                        <p>
                          <span className="font-medium">Total de docks:</span> {station.totalDocks}
                        </p>
                        <p>
                          <span className="font-medium">Ocupación:</span> {Math.round((station.availableBikes / station.totalDocks) * 100)}%
                        </p>
                      </div>
                    </div>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          )}
        </div>
      </div>

      {/* Estadísticas del mapa */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-600 mb-1">Estaciones Activas</h4>
          <p className="text-2xl font-bold text-green-600">
            {stations.filter(s => s.status === 'active').length}
          </p>
        </div>
        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-600 mb-1">En Mantenimiento</h4>
          <p className="text-2xl font-bold text-yellow-600">
            {stations.filter(s => s.status === 'maintenance').length}
          </p>
        </div>
        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-600 mb-1">Bicis Disponibles</h4>
          <p className="text-2xl font-bold text-blue-600">
            {stations.reduce((sum, station) => sum + station.availableBikes, 0)}
          </p>
        </div>
      </div>
    </div>
  )
} 