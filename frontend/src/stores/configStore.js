import { create } from 'zustand'

const useConfigStore = create((set, get) => ({
  sessionId: 'default-' + Date.now(),
  config: {
    llm: { provider: 'openai_compatible', base_url: '', api_key: '', model: 'gpt-4o', temperature: 0.7, max_tokens: 4096 },
    embedding: { provider: 'openai', base_url: '', api_key: '', model: 'text-embedding-3-small', dimensions: 1536 },
    retrieval: { mode: 'hybrid', top_k: 20, hybrid: { dense_weight: 0.7, sparse_weight: 0.3 } },
    reranker: { enabled: true, provider: 'bge', top_k: 5 },
    chunking: { strategy: 'semantic', semantic: { max_chunk_size: 1024, min_chunk_size: 100, similarity_threshold: 0.75 }, recursive: { chunk_size: 512, chunk_overlap: 50 }, fixed: { chunk_size: 512, chunk_overlap: 50 } },
    agent: { strategy: 'react', max_iterations: 10 },
    multimodal: { ocr: { enabled: true, provider: 'tesseract' }, asr: { enabled: true, provider: 'whisper' } },
    conversation: { max_history_turns: 20, compression: { strategy: 'sliding_window', window_size: 10 }, coreference_resolution: true },
  },
  messages: [],
  isLoading: false,
  wsConnection: null,

  setConfig: (path, value) => set((state) => {
    const newConfig = { ...state.config }
    const keys = path.split('.')
    let obj = newConfig
    for (let i = 0; i < keys.length - 1; i++) {
      if (!obj[keys[i]]) obj[keys[i]] = {}
      obj = obj[keys[i]]
    }
    obj[keys[keys.length - 1]] = value
    return { config: newConfig }
  }),

  addMessage: (msg) => set((state) => ({
    messages: [...state.messages, msg]
  })),

  setLoading: (loading) => set({ isLoading: loading }),
  clearMessages: () => set({ messages: [] }),
  setSessionId: (id) => set({ sessionId: id }),
}))

export default useConfigStore