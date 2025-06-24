import React from 'react'
import type { Metadata } from 'next'
import './globals.css'
import Script from 'next/script'

export const metadata: Metadata = {
  title: '🚴 BA Bicis Dashboard',
  description: 'Dashboard para visualización y predicciones de BA Bicis en tiempo real',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es">
      <head>
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossOrigin=""
        />
      </head>
      <body className="bg-gray-50 min-h-screen">
        <header className="bg-ba-blue text-white shadow-lg">
          <div className="container mx-auto px-4 py-6">
            <h1 className="text-3xl font-bold">🚴 BA Bicis Dashboard</h1>
            <p className="text-blue-100 mt-2">Predicciones y análisis en tiempo real</p>
          </div>
        </header>
        <main>{children}</main>
        <footer className="bg-gray-800 text-white py-8 mt-12">
          <div className="container mx-auto px-4 text-center">
            <p>&copy; {new Date().getFullYear()} BA Bicis Dashboard - Proyecto Final ML</p>
          </div>
        </footer>
        
        {/* Scripts para Leaflet */}
        <Script
          src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossOrigin=""
          strategy="beforeInteractive"
        />
        <Script
          src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"
          strategy="beforeInteractive"
        />
      </body>
    </html>
  )
} 