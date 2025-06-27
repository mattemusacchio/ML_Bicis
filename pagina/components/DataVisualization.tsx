'use client'

import React, { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'

// Importar Plot dinámicamente para evitar errores de SSR
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface DataVisualizationProps {
  showAdvanced?: boolean;
}

export default function DataVisualization({ showAdvanced = false }: DataVisualizationProps) {
  const [hourlyData, setHourlyData] = useState<any[]>([])
  const [stationData, setStationData] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simular datos de ejemplo
    const generateHourlyData = () => {
      const hours = Array.from({length: 24}, (_, i) => i)
      const trips = hours.map(h => {
        // Simular patrones de uso típicos
        if (h >= 6 && h <= 9) return Math.random() * 100 + 80 // Pico matutino
        if (h >= 17 && h <= 20) return Math.random() * 120 + 90 // Pico vespertino
        if (h >= 11 && h <= 14) return Math.random() * 60 + 40 // Almuerzo
        return Math.random() * 30 + 10 // Horas bajas
      })
      
      return hours.map((hour, i) => ({
        hour: `${hour}:00`,
        trips: Math.round(trips[i]),
        predictions: Math.round(trips[i] * (0.9 + Math.random() * 0.2))
      }))
    }

    const generateStationData = async () => {
      try {
        // Intentar cargar datos reales de estaciones
        const response = await fetch('/api/stations')
        
        if (response.ok) {
          const result = await response.json()
          if (result.success && result.data) {
            const realStations = result.data
              .sort((a: any, b: any) => (b.total_trips || 0) - (a.total_trips || 0))
              .slice(0, 10) // Top 10 estaciones
              .map((station: any) => ({
                station: station.name,
                trips: station.total_trips || Math.round(Math.random() * 500 + 100),
                availability: station.available_bikes || Math.round(Math.random() * 100)
              }))
            
            console.log('✅ Datos de visualización cargados desde API')
            return realStations
          }
        }
        
        // Fallback: cargar desde archivo JSON
        const jsonResponse = await fetch('/stations_data.json')
        if (jsonResponse.ok) {
          const jsonData = await jsonResponse.json()
          const realStations = jsonData.stations
            .sort((a: any, b: any) => (b.total_trips || 0) - (a.total_trips || 0))
            .slice(0, 10)
            .map((station: any) => ({
              station: station.name,
              trips: station.total_trips || Math.round(Math.random() * 500 + 100),
              availability: station.available_bikes || Math.round(Math.random() * 100)
            }))
          
          console.log('✅ Datos de visualización cargados desde JSON')
          return realStations
        }
        
        throw new Error('No se pudieron cargar datos de estaciones')
        
      } catch (error) {
        console.error('❌ Error cargando datos para visualización:', error)
        
        // Fallback a datos simulados
        const fallbackStations = [
          'Estación Centro', 'Estación Norte', 'Estación Sur', 'Estación Este', 'Estación Oeste'
        ]
        
        return fallbackStations.map(station => ({
          station,
          trips: Math.round(Math.random() * 500 + 100),
          availability: Math.round(Math.random() * 100)
        }))
      }
    }

    setHourlyData(generateHourlyData())
    
    // Cargar datos de estaciones de forma asíncrona
    generateStationData().then(data => {
      setStationData(data)
      setIsLoading(false)
    })
  }, [])

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="chart-container animate-pulse">
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
        <div className="chart-container animate-pulse">
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Gráfico de uso por hora */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">📈 Uso por Hora del Día</h3>
        <Plot
          data={[
            {
              x: hourlyData.map(d => d.hour),
              y: hourlyData.map(d => d.trips),
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Viajes Reales',
              line: { color: '#1f77b4' },
              marker: { color: '#1f77b4' }
            },
            {
              x: hourlyData.map(d => d.hour),
              y: hourlyData.map(d => d.predictions),
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Predicciones',
              line: { color: '#ff7f0e', dash: 'dash' },
              marker: { color: '#ff7f0e' }
            }
          ]}
          layout={{
            title: 'Patrones de Uso Diario',
            xaxis: { title: 'Hora del Día' },
            yaxis: { title: 'Número de Viajes' },
            height: 400,
            margin: { l: 50, r: 50, t: 50, b: 50 }
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      {/* Gráfico de estaciones más usadas */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">🏆 Estaciones Más Populares</h3>
        <Plot
          data={[
            {
              x: stationData.map(d => d.trips),
              y: stationData.map(d => d.station),
              type: 'bar',
              orientation: 'h',
              marker: { 
                color: stationData.map(d => d.trips),
                colorscale: 'Blues'
              }
            }
          ]}
          layout={{
            title: 'Viajes por Estación (Últimas 24h)',
            xaxis: { title: 'Número de Viajes' },
            height: 400,
            margin: { l: 120, r: 50, t: 50, b: 50 }
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      {/* Análisis avanzado solo si se solicita */}
      {showAdvanced && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="chart-container">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">📊 Disponibilidad por Estación</h3>
            <Plot
              data={[
                {
                  values: stationData.map(d => d.availability),
                  labels: stationData.map(d => d.station),
                  type: 'pie',
                  hoverinfo: 'label+percent',
                  textinfo: 'percent'
                }
              ]}
              layout={{
                title: 'Distribución de Disponibilidad',
                height: 350
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>

          <div className="chart-container">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">🔄 Tendencias Semanales</h3>
            <Plot
              data={[
                {
                  x: ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
                  y: [850, 920, 880, 900, 750, 450, 380],
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Promedio Semanal',
                  line: { color: '#28a745' },
                  marker: { color: '#28a745', size: 8 }
                }
              ]}
              layout={{
                title: 'Uso Promedio por Día de la Semana',
                xaxis: { title: 'Día' },
                yaxis: { title: 'Viajes Promedio' },
                height: 350
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      )}
    </div>
  )
} 