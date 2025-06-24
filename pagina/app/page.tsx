'use client'

import { useState, useEffect } from 'react'
import DataVisualization from '../components/DataVisualization'
import PredictionInterface from '../components/PredictionInterface'
import HeatmapMap from '../components/HeatmapMap'
import RealTimeMetrics from '../components/RealTimeMetrics'

export default function Home() {
  const [activeTab, setActiveTab] = useState('overview')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simular carga inicial
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 1000)
    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Cargando dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Navegación por pestañas */}
      <div className="bg-white rounded-lg shadow-md mb-8">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {[
              { id: 'overview', name: '📊 Resumen', icon: '📊' },
              { id: 'predictions', name: '🔮 Predicciones', icon: '🔮' },
              { id: 'map', name: '🗺️ Mapa', icon: '🗺️' },
              { id: 'analytics', name: '📈 Análisis', icon: '📈' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'border-ba-blue text-ba-blue'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.name}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Contenido de las pestañas */}
      <div className="space-y-8">
        {activeTab === 'overview' && (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">📊 Resumen General</h2>
            <RealTimeMetrics />
            <div className="mt-8">
              <DataVisualization />
            </div>
          </div>
        )}

        {activeTab === 'predictions' && (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">🔮 Predicciones en Tiempo Real</h2>
            <PredictionInterface />
          </div>
        )}

        {activeTab === 'map' && (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">🔥 Mapa de Calor de Viajes</h2>
            <HeatmapMap />
          </div>
        )}

        {activeTab === 'analytics' && (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-6">📈 Análisis Avanzado</h2>
            <DataVisualization showAdvanced={true} />
          </div>
        )}
      </div>
    </div>
  )
} 