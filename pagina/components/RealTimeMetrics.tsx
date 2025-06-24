'use client'

import React, { useState, useEffect } from 'react'

interface MetricData {
  totalStations: number;
  activeBikes: number;
  totalTrips: number;
  avgWaitTime: number;
}

export default function RealTimeMetrics() {
  const [metrics, setMetrics] = useState<MetricData>({
    totalStations: 0,
    activeBikes: 0,
    totalTrips: 0,
    avgWaitTime: 0
  })
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simular datos en tiempo real
    const fetchMetrics = () => {
      setMetrics({
        totalStations: 402,
        activeBikes: 3847,
        totalTrips: 15203,
        avgWaitTime: 3.2
      })
      setIsLoading(false)
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 30000) // Actualizar cada 30 segundos

    return () => clearInterval(interval)
  }, [])

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="metric-card animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    )
  }

  const metricsData = [
    {
      title: 'Estaciones Totales',
      value: metrics.totalStations.toLocaleString(),
      icon: '🚉',
      color: 'text-blue-600'
    },
    {
      title: 'Bicis Activas',
      value: metrics.activeBikes.toLocaleString(),
      icon: '🚴',
      color: 'text-green-600'
    },
    {
      title: 'Viajes Hoy',
      value: metrics.totalTrips.toLocaleString(),
      icon: '📊',
      color: 'text-purple-600'
    },
    {
      title: 'Tiempo Espera Prom.',
      value: `${metrics.avgWaitTime} min`,
      icon: '⏱️',
      color: 'text-orange-600'
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metricsData.map((metric, index) => (
        <div key={index} className="metric-card hover:shadow-lg transition-shadow">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600 mb-1">{metric.title}</p>
              <p className={`text-2xl font-bold ${metric.color}`}>{metric.value}</p>
            </div>
            <div className="text-3xl">{metric.icon}</div>
          </div>
        </div>
      ))}
    </div>
  )
} 