'use client'

import React, { useState, useEffect, useRef } from 'react'
import dynamic from 'next/dynamic'

// Importar componentes de mapa dinámicamente
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
)
const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
)

interface HeatmapPoint {
  lat: number;
  lng: number;
  intensity: number;
  type: string;
  station: string;
  duration: number;
}

interface HeatmapStats {
  totalTrips: number;
  topStations: Array<{
    station: string;
    count: number;
    type: string;
  }>;
  processedRecords: number;
}

export default function HeatmapMap() {
  const [heatmapData, setHeatmapData] = useState<HeatmapPoint[]>([])
  const [stats, setStats] = useState<HeatmapStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [dataSource, setDataSource] = useState('')
  const [filterType, setFilterType] = useState<'all' | 'origen' | 'destino'>('all')
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [mapStyle, setMapStyle] = useState('dark')

  useEffect(() => {
    fetchHeatmapData()
  }, [])

  const fetchHeatmapData = async () => {
    try {
      const response = await fetch('/api/heatmap')
      const result = await response.json()
      
      if (result.success) {
        setHeatmapData(result.data)
        setStats(result.stats)
        setDataSource(result.source)
      }
    } catch (error) {
      console.error('Error cargando datos del heatmap:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const filteredData = heatmapData.filter(point => 
    filterType === 'all' || point.type === filterType
  )

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow-md p-6 animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="h-96 bg-gray-200 rounded"></div>
        </div>
      </div>
    )
  }

  const mapTileStyles = {
    dark: {
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      attribution: '&copy; <a href="https://carto.com/attributions">CARTO</a>',
      name: '🌙 Oscuro'
    },
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: '&copy; <a href="https://www.esri.com/">Esri</a>',
      name: '🛰️ Satélite'
    },
    streets: {
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      name: '🗺️ Calles'
    }
  }

  return (
    <div className="space-y-6">
      {/* Panel de control */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              🔥 Mapa de Calor - BA Bicis
            </h3>
            <div className="flex flex-wrap gap-4 text-sm text-gray-600">
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                dataSource === 'real' ? 'bg-green-100 text-green-800' : 
                dataSource === 'simulated' ? 'bg-blue-100 text-blue-800' :
                'bg-yellow-100 text-yellow-800'
              }`}>
                {dataSource === 'real' ? '📊 Datos Reales' : 
                 dataSource === 'simulated' ? '🎯 Datos Simulados' : 
                 '⚠️ Modo Fallback'}
              </span>
              {stats && (
                <>
                  <span className="bg-gray-100 px-2 py-1 rounded-full text-xs">
                    🚴 {stats.totalTrips.toLocaleString()} viajes
                  </span>
                  <span className="bg-gray-100 px-2 py-1 rounded-full text-xs">
                    📍 {filteredData.length.toLocaleString()} puntos
                  </span>
                </>
              )}
            </div>
          </div>
          
          <div className="flex flex-wrap gap-2">
            {/* Filtro de tipo */}
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-ba-blue focus:border-ba-blue"
            >
              <option value="all">🔄 Todos</option>
              <option value="origen">🚀 Orígenes</option>
              <option value="destino">🎯 Destinos</option>
            </select>
            
            {/* Estilo de mapa */}
            <select
              value={mapStyle}
              onChange={(e) => setMapStyle(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-ba-blue focus:border-ba-blue"
            >
              {Object.entries(mapTileStyles).map(([key, style]) => (
                <option key={key} value={key}>{style.name}</option>
              ))}
            </select>
            
            {/* Toggle heatmap */}
            <button
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                showHeatmap 
                  ? 'bg-red-500 text-white hover:bg-red-600'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {showHeatmap ? '🔥 Calor ON' : '🔥 Calor OFF'}
            </button>
          </div>
        </div>
      </div>

      {/* Mapa principal */}
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="h-[600px] relative">
          {typeof window !== 'undefined' && (
            <MapContainer
              center={[-34.6037, -58.3816]}
              zoom={12}
              style={{ height: '100%', width: '100%' }}
              className="z-0"
            >
              <TileLayer
                url={mapTileStyles[mapStyle as keyof typeof mapTileStyles].url}
                attribution={mapTileStyles[mapStyle as keyof typeof mapTileStyles].attribution}
              />
              
              <MapHandler 
                data={filteredData}
                showHeatmap={showHeatmap}
                filterType={filterType}
              />
            </MapContainer>
          )}
        </div>
      </div>

      {/* Panel de estadísticas */}
      {stats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Ranking de estaciones de origen */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              🚀 <span className="ml-2">Top Estaciones de Origen</span>
            </h4>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {stats.topStations
                .filter(s => s.type === 'origen')
                .slice(0, 10)
                .map((station, index) => (
                <div key={index} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
                  <div className="flex items-center space-x-3">
                    <span className="text-xl">
                      {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🚉'}
                    </span>
                    <div>
                      <span className="text-sm font-medium text-gray-900 block">
                        {station.station.length > 30 ? 
                          `${station.station.substring(0, 30)}...` : 
                          station.station
                        }
                      </span>
                      <span className="text-xs text-gray-500">#{index + 1}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-bold text-ba-blue">
                      {station.count.toLocaleString()}
                    </span>
                    <span className="text-xs text-gray-500 block">viajes</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Ranking de estaciones de destino */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              🎯 <span className="ml-2">Top Estaciones de Destino</span>
            </h4>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {stats.topStations
                .filter(s => s.type === 'destino')
                .slice(0, 10)
                .map((station, index) => (
                <div key={index} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
                  <div className="flex items-center space-x-3">
                    <span className="text-xl">
                      {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '🏁'}
                    </span>
                    <div>
                      <span className="text-sm font-medium text-gray-900 block">
                        {station.station.length > 30 ? 
                          `${station.station.substring(0, 30)}...` : 
                          station.station
                        }
                      </span>
                      <span className="text-xs text-gray-500">#{index + 1}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-bold text-green-600">
                      {station.count.toLocaleString()}
                    </span>
                    <span className="text-xs text-gray-500 block">arribos</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Leyenda mejorada */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">📖 Guía del Mapa de Calor</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-4 h-4 bg-red-600 rounded-full opacity-80"></div>
            <span className="text-sm text-gray-700">Muy alta actividad</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-4 h-4 bg-orange-500 rounded-full opacity-80"></div>
            <span className="text-sm text-gray-700">Alta actividad</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-4 h-4 bg-yellow-500 rounded-full opacity-80"></div>
            <span className="text-sm text-gray-700">Actividad moderada</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-4 h-4 bg-blue-500 rounded-full opacity-80"></div>
            <span className="text-sm text-gray-700">Actividad baja</span>
          </div>
        </div>
        <div className="bg-gray-50 p-3 rounded-md">
          <p className="text-xs text-gray-600 leading-relaxed">
            🔍 <strong>Cómo interpretar:</strong> Las zonas más rojizas indican mayor concentración de viajes. 
            Usa los filtros para ver solo orígenes o destinos. El mapa muestra datos 
            {dataSource === 'real' ? ' reales extraídos de tu dataset' : ' simulados basados en patrones típicos'}.
            Haz clic en los puntos para ver detalles específicos de cada zona.
          </p>
        </div>
      </div>
    </div>
  )
}

// Componente para manejar el mapa y heatmap
function MapHandler({ data, showHeatmap, filterType }: { 
  data: HeatmapPoint[], 
  showHeatmap: boolean,
  filterType: string 
}) {
  const mapRef = useRef<any>(null)

  useEffect(() => {
    if (typeof window === 'undefined' || !data.length) return

    const timer = setTimeout(() => {
      // Buscar el mapa en el DOM
      const mapElement = document.querySelector('.leaflet-container')
      if (!mapElement) return

      // Obtener la instancia del mapa
      const map = (mapElement as any)._leaflet_map
      if (!map) return

      mapRef.current = map

      // Limpiar layers anteriores
      map.eachLayer((layer: any) => {
        if (layer.options && (layer.options.isHeatmapLayer || layer.options.isMarkerLayer)) {
          map.removeLayer(layer)
        }
      })

      if (showHeatmap && data.length > 0) {
        // Preparar datos para el heatmap
        const heatData = data.map(point => [
          point.lat,
          point.lng,
          Math.min(point.intensity * 0.8, 1)
        ])

        // Crear heatmap usando leaflet.heat
        if ((window as any).L && (window as any).L.heatLayer) {
          const heat = (window as any).L.heatLayer(heatData, {
            radius: 25,
            blur: 15,
            maxZoom: 17,
            max: 1.0,
            gradient: {
              0.0: 'blue',
              0.2: 'cyan', 
              0.4: 'lime',
              0.6: 'yellow',
              0.8: 'orange',
              1.0: 'red'
            },
            isHeatmapLayer: true
          })

          heat.addTo(map)
        }

        // Agregar marcadores para top estaciones
        const stationCounts = new Map()
        data.forEach(point => {
          const count = stationCounts.get(point.station) || 0
          stationCounts.set(point.station, count + 1)
        })

        const topStations = Array.from(stationCounts.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)

        topStations.forEach(([stationName, count], index) => {
          const stationPoints = data.filter(p => p.station === stationName)
          if (stationPoints.length > 0) {
            const point = stationPoints[0]
            
            const marker = (window as any).L.marker([point.lat, point.lng], {
              icon: (window as any).L.divIcon({
                html: `<div style="background: white; border: 2px solid #1f77b4; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${index + 1}</div>`,
                iconSize: [30, 30],
                className: 'custom-div-icon'
              }),
              isMarkerLayer: true
            })

            marker.bindPopup(`
              <div style="padding: 12px; min-width: 250px;">
                <h4 style="font-weight: bold; margin-bottom: 8px; color: #1f2937;">🏆 ${stationName}</h4>
                <div style="font-size: 14px; line-height: 1.4;">
                  <p><strong>Ranking:</strong> #${index + 1}</p>
                  <p><strong>Total de puntos:</strong> ${count.toLocaleString()}</p>
                  <p><strong>Tipo:</strong> ${point.type === 'origen' ? '🚀 Origen' : '🎯 Destino'}</p>
                  <p><strong>Duración prom.:</strong> ${Math.round(point.duration / 60)} min</p>
                </div>
              </div>
            `)

            marker.addTo(map)
          }
        })
      }

    }, 1000) // Esperar a que el mapa se inicialice

    return () => clearTimeout(timer)
  }, [data, showHeatmap, filterType])

  return null
} 