import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

// Chat
export const sendMessage = (data) => api.post('/chat/', data)
export const getHealth = () => api.get('/health')

// Documents
export const uploadDocument = (file, sessionId = 'default') => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post(`/documents/upload?session_id=${sessionId}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}
export const getDocStats = (sessionId = 'default') =>
  api.get(`/documents/stats?session_id=${sessionId}`)

// Config
export const getCurrentConfig = (sessionId = 'default') =>
  api.get(`/config/current?session_id=${sessionId}`)
export const updateConfig = (data) => api.post('/config/update', data)
export const getComponents = () => api.get('/config/components')
export const resetConfig = (sessionId = 'default') =>
  api.post(`/config/reset?session_id=${sessionId}`)

// Evaluation
export const listDatasets = (sessionId = 'default') =>
  api.get(`/evaluation/datasets?session_id=${sessionId}`)
export const runEvaluation = (data) => api.post('/evaluation/run', data)

// WebSocket
export const createChatWS = (sessionId) => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return new WebSocket(`${protocol}//${window.location.host}/api/chat/ws/${sessionId}`)
}

export default api