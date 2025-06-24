#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚴 Configurando BA Bicis Dashboard...\n');

// Verificar que estamos en la carpeta correcta
if (!fs.existsSync('package.json')) {
  console.error('❌ Error: Ejecuta este script desde la carpeta "pagina"');
  process.exit(1);
}

try {
  // 1. Instalar dependencias
  console.log('📦 Instalando dependencias...');
  execSync('npm install', { stdio: 'inherit' });
  console.log('✅ Dependencias instaladas correctamente\n');

  // 2. Crear archivo de variables de entorno si no existe
  const envPath = '.env.local';
  if (!fs.existsSync(envPath)) {
    console.log('🔧 Creando archivo de configuración...');
    const envContent = `# Configuración de BA Bicis Dashboard
NEXT_PUBLIC_API_URL=http://localhost:3000/api
NODE_ENV=development

# Configuraciones opcionales
# MODEL_PATH=/path/to/your/model
# DATABASE_URL=your_database_url
`;
    fs.writeFileSync(envPath, envContent);
    console.log('✅ Archivo .env.local creado\n');
  }

  // 3. Verificar estructura de carpetas
  console.log('📁 Verificando estructura de proyecto...');
  const requiredDirs = ['app', 'components', 'app/api'];
  requiredDirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`✅ Carpeta ${dir} creada`);
    }
  });

  // 4. Verificar archivos importantes
  const requiredFiles = [
    'app/layout.tsx',
    'app/page.tsx',
    'app/globals.css',
    'tailwind.config.js',
    'next.config.js'
  ];
  
  let missingFiles = [];
  requiredFiles.forEach(file => {
    if (!fs.existsSync(file)) {
      missingFiles.push(file);
    }
  });

  if (missingFiles.length > 0) {
    console.log('⚠️  Archivos faltantes detectados:');
    missingFiles.forEach(file => console.log(`   - ${file}`));
    console.log('   Por favor, asegúrate de que todos los archivos estén presentes.\n');
  } else {
    console.log('✅ Todos los archivos necesarios están presentes\n');
  }

  // 5. Crear script de inicio rápido
  const startScript = `#!/bin/bash
echo "🚴 Iniciando BA Bicis Dashboard..."
echo "🌐 La aplicación estará disponible en http://localhost:3000"
echo "🔧 Modo desarrollo - Los cambios se reflejarán automáticamente"
echo ""
npm run dev
`;

  fs.writeFileSync('start.sh', startScript);
  if (process.platform !== 'win32') {
    execSync('chmod +x start.sh');
  }
  console.log('✅ Script de inicio creado (start.sh)\n');

  // 6. Información final
  console.log('🎉 ¡Configuración completada exitosamente!\n');
  console.log('📋 Próximos pasos:');
  console.log('   1. Para desarrollo local: npm run dev');
  console.log('   2. Para producción: npm run build && npm start');
  console.log('   3. Para desplegar en Vercel: vercel');
  console.log('');
  console.log('🌐 La aplicación estará disponible en: http://localhost:3000');
  console.log('📖 Lee el README.md para más información');
  console.log('');
  console.log('🚀 ¡Listo para comenzar!');

} catch (error) {
  console.error('❌ Error durante la configuración:', error.message);
  console.log('\n🔧 Soluciones posibles:');
  console.log('   1. Verifica tu conexión a internet');
  console.log('   2. Asegúrate de tener Node.js >= 18 instalado');
  console.log('   3. Ejecuta: npm cache clean --force');
  console.log('   4. Intenta nuevamente: node setup.js');
  process.exit(1);
} 