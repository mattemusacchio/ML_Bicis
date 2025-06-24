import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const { stationId, dateTime } = await request.json()

    // Validar entrada
    if (!stationId || !dateTime) {
      return NextResponse.json(
        { error: 'stationId y dateTime son requeridos' },
        { status: 400 }
      )
    }

    // Simular llamada al modelo ML
    // En un caso real, aquí cargarías tu modelo y harías la predicción
    const prediction = await simulateModelPrediction(stationId, dateTime)

    return NextResponse.json({
      success: true,
      data: prediction
    })
  } catch (error) {
    console.error('Error en predicción:', error)
    return NextResponse.json(
      { error: 'Error interno del servidor' },
      { status: 500 }
    )
  }
}

async function simulateModelPrediction(stationId: string, dateTime: string) {
  // Simular tiempo de procesamiento
  await new Promise(resolve => setTimeout(resolve, 1000))

  const baseArrival = Math.random() * 50 + 10
  const timeOfDay = new Date(dateTime).getHours()
  
  // Ajustar predicción según hora del día
  let adjustedArrival = baseArrival
  if (timeOfDay >= 7 && timeOfDay <= 9) adjustedArrival *= 1.5 // Pico matutino
  if (timeOfDay >= 17 && timeOfDay <= 19) adjustedArrival *= 1.8 // Pico vespertino
  if (timeOfDay >= 23 || timeOfDay <= 6) adjustedArrival *= 0.3 // Noche

  return {
    stationId,
    predicted: Math.round(adjustedArrival),
    confidence: Math.round((Math.random() * 30 + 70) * 100) / 100,
    timestamp: new Date().toISOString(),
    factors: {
      timeOfDay: timeOfDay,
      dayOfWeek: new Date(dateTime).getDay(),
      weather: 'clear' // Simulado
    }
  }
} 