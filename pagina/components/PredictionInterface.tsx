'use client'

import React, { useState, useEffect } from 'react'

interface PredictionData {
  stationId: string;
  stationName: string;
  predicted: number;
  confidence: number;
}

export default function PredictionInterface() {
  const [selectedStation, setSelectedStation] = useState('')
  const [selectedTime, setSelectedTime] = useState('')
  const [predictions, setPredictions] = useState<PredictionData[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [stations, setStations] = useState<{id: string, name: string}[]>([])
  const [stationsLoading, setStationsLoading] = useState(true)

  // Cargar estaciones reales al montar el componente
  useEffect(() => {
    loadRealStations()
  }, [])

  const loadRealStations = async () => {
    try {
      setStationsLoading(true)
      
      // Intentar cargar desde la API primero
      const response = await fetch('/api/stations')
      
      if (response.ok) {
        const result = await response.json()
        if (result.success && result.data) {
          const stationsList = result.data
            .sort((a: any, b: any) => (b.total_trips || 0) - (a.total_trips || 0)) // Ordenar por popularidad
            .slice(0, 50) // Solo top 50 para el dropdown
            .map((station: any) => ({
              id: station.id,
              name: station.name
            }))
          
          setStations(stationsList)
          console.log(`✅ Cargadas ${stationsList.length} estaciones para predicción`)
          return
        }
      }
      
      // Fallback: cargar desde archivo JSON estático
      const jsonResponse = await fetch('/stations_data.json')
      if (jsonResponse.ok) {
        const jsonData = await jsonResponse.json()
        const stationsList = jsonData.stations
          .sort((a: any, b: any) => (b.total_trips || 0) - (a.total_trips || 0))
          .slice(0, 50)
          .map((station: any) => ({
            id: station.id,
            name: station.name
          }))
        
        setStations(stationsList)
        console.log(`✅ Cargadas ${stationsList.length} estaciones desde JSON`)
        return
      }
      
      throw new Error('No se pudieron cargar las estaciones')
      
    } catch (error) {
      console.error('❌ Error cargando estaciones:', error)
      
      // Fallback a estaciones básicas
      const fallbackStations = [
        { id: '1', name: 'Estación Centro' },
        { id: '2', name: 'Estación Norte' },
        { id: '3', name: 'Estación Sur' }
      ]
      setStations(fallbackStations)
      console.log('⚠️ Usando estaciones de fallback')
    } finally {
      setStationsLoading(false)
    }
  }

  const handlePredict = async () => {
    if (!selectedStation || !selectedTime) {
      alert('Por favor selecciona una estación y un horario')
      return
    }

    setIsLoading(true)

    // Simular llamada a API de predicción
    setTimeout(() => {
      const selectedStationData = stations.find(s => s.id === selectedStation)
      const mockPrediction: PredictionData = {
        stationId: selectedStation,
        stationName: selectedStationData?.name || 'Desconocida',
        predicted: Math.round(Math.random() * 50 + 10),
        confidence: Math.round((Math.random() * 30 + 70) * 100) / 100
      }
      
      setPredictions([mockPrediction, ...predictions.slice(0, 4)])
      setIsLoading(false)
    }, 2000)
  }

  const getCurrentDateTime = () => {
    const now = new Date()
    now.setMinutes(now.getMinutes() + 30) // 30 minutos en el futuro
    return now.toISOString().slice(0, 16)
  }

  return (
    <div className="space-y-8">
      {/* Interfaz de predicción */}
      <div className="prediction-box">
        <h3 className="text-xl font-semibold text-gray-900 mb-6">🔮 Hacer Predicción</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Seleccionar Estación
            </label>
            <select
              value={selectedStation}
              onChange={(e) => setSelectedStation(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-ba-blue focus:border-ba-blue"
            >
              <option value="">Elegir estación...</option>
              {stations.map((station) => (
                <option key={station.id} value={station.id}>
                  {station.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Fecha y Hora
            </label>
            <input
              type="datetime-local"
              value={selectedTime}
              onChange={(e) => setSelectedTime(e.target.value)}
              min={getCurrentDateTime()}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-ba-blue focus:border-ba-blue"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={handlePredict}
              disabled={isLoading || !selectedStation || !selectedTime}
              className="w-full bg-ba-blue text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="loading-spinner mr-2"></div>
                  Prediciendo...
                </div>
              ) : (
                '🚀 Predecir Arribos'
              )}
            </button>
          </div>
        </div>

        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <span className="text-yellow-400">⚠️</span>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-800">
                <strong>Nota:</strong> Las predicciones se basan en datos históricos y patrones de uso. 
                Los resultados son estimaciones y pueden variar según condiciones externas.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Resultados de predicciones */}
      {predictions.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">📊 Resultados de Predicciones</h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Estación
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Arribos Predichos
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confianza
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Estado
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {predictions.map((prediction, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {prediction.stationName}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {prediction.predicted} arribos
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div 
                            className="bg-green-600 h-2 rounded-full" 
                            style={{ width: `${prediction.confidence}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-600">{prediction.confidence}%</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        prediction.predicted > 30 
                          ? 'bg-green-100 text-green-800' 
                          : prediction.predicted > 15 
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {prediction.predicted > 30 ? '🟢 Alta demanda' : 
                         prediction.predicted > 15 ? '🟡 Demanda media' : '🔴 Baja demanda'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
} 