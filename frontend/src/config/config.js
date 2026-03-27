// Vite uses import.meta.env, Create React App uses process.env

const getEnvVar = (key, defaultValue) => {
  if (typeof import.meta !== 'undefined' && import.meta.env) {
    return import.meta.env[key] || defaultValue;
  }
  if (typeof process !== 'undefined' && process.env) {
    return process.env[key] || defaultValue;
  }
  return defaultValue;
};

export const config = {
  api: {
    baseURL: getEnvVar('VITE_API_URL', 'http://localhost:5000'),
    timeout: parseInt(getEnvVar('VITE_API_TIMEOUT', '60000')),
  },
  upload: {
    maxFileSize: parseInt(getEnvVar('VITE_MAX_FILE_SIZE', '33554432')), // 32MB
    allowedImageTypes: ['image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'],
    // Heart analysis also accepts .dat, .hea, .csv, .edf files
    allowedHeartTypes: [
      'image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp',
      'application/octet-stream', // .dat files
      'text/plain', // .hea files
      'text/csv', // .csv files
      '.dat', '.hea', '.csv', '.edf', '.mat'
    ],
    allowedHeartExtensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'dat', 'hea', 'csv', 'edf', 'mat'],
  },
  features: {
    analytics: getEnvVar('VITE_ENABLE_ANALYTICS', 'false') === 'true',
    caching: getEnvVar('VITE_ENABLE_CACHING', 'true') === 'true',
  },
  app: {
    version: getEnvVar('VITE_APP_VERSION', '1.0.0'),
    name: getEnvVar('VITE_APP_NAME', 'MediDiagnose'),
  },
};

export default config;