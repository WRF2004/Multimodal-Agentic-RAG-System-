import React, { useState, useEffect } from 'react'
import { Card, Form, Input, Select, Slider, Switch, InputNumber, Button, Tabs, message, Divider, Row, Col, Typography } from 'antd'
import { SaveOutlined, ReloadOutlined } from '@ant-design/icons'
import useConfigStore from '../stores/configStore'
import { updateConfig, resetConfig, getComponents } from '../api/client'

const { Option } = Select
const { Title, Text } = Typography

const ConfigPanel = () => {
  const { sessionId, config, setConfig } = useConfigStore()
  const [components, setComponents] = useState({})
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getComponents().then(res => setComponents(res.data)).catch(() => {})
  }, [])

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateConfig({ session_id: sessionId, overrides: config })
      message.success('配置已保存并实时生效!')
    } catch (err) {
      message.error('保存失败: ' + (err.response?.data?.detail || err.message))
    }
    setSaving(false)
  }

  const handleReset = async () => {
    await resetConfig(sessionId)
    message.success('配置已重置为默认值')
    window.location.reload()
  }

  const tabItems = [
    {
      key: 'llm',
      label: '🤖 对话模型',
      children: (
        <Card size="small">
          <Form layout="vertical">
            <Form.Item label="API 地址 (base_url)">
              <Input value={config.llm?.base_url} onChange={e => setConfig('llm.base_url', e.target.value)} placeholder="https://api.openai.com/v1" />
            </Form.Item>
            <Form.Item label="API Key">
              <Input.Password value={config.llm?.api_key} onChange={e => setConfig('llm.api_key', e.target.value)} placeholder="sk-..." />
            </Form.Item>
            <Form.Item label="模型名称">
              <Input value={config.llm?.model} onChange={e => setConfig('llm.model', e.target.value)} placeholder="gpt-4o" />
            </Form.Item>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label={`温度: ${config.llm?.temperature || 0.7}`}>
                  <Slider min={0} max={2} step={0.1} value={config.llm?.temperature || 0.7} onChange={v => setConfig('llm.temperature', v)} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="最大Token">
                  <InputNumber min={256} max={128000} value={config.llm?.max_tokens || 4096} onChange={v => setConfig('llm.max_tokens', v)} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
          </Form>
        </Card>
      )
    },
    {
      key: 'embedding',
      label: '📐 Embedding',
      children: (
        <Card size="small">
          <Form layout="vertical">
            <Form.Item label="Provider">
              <Select value={config.embedding?.provider} onChange={v => setConfig('embedding.provider', v)}>
                <Option value="openai">OpenAI</Option>
                <Option value="sentence_transformer">SentenceTransformer (本地)</Option>
              </Select>
            </Form.Item>
            <Form.Item label="API 地址">
              <Input value={config.embedding?.base_url} onChange={e => setConfig('embedding.base_url', e.target.value)} />
            </Form.Item>
            <Form.Item label="API Key">
              <Input.Password value={config.embedding?.api_key} onChange={e => setConfig('embedding.api_key', e.target.value)} />
            </Form.Item>
            <Form.Item label="模型">
              <Input value={config.embedding?.model} onChange={e => setConfig('embedding.model', e.target.value)} />
            </Form.Item>
          </Form>
        </Card>
      )
    },
    {
      key: 'retrieval',
      label: '🔍 检索',
      children: (
        <Card size="small">
          <Form layout="vertical">
            <Form.Item label="检索模式">
              <Select value={config.retrieval?.mode} onChange={v => setConfig('retrieval.mode', v)}>
                <Option value="dense">仅稠密 (向量)</Option>
                <Option value="sparse">仅稀疏 (BM25)</Option>
                <Option value="hybrid">混合检索</Option>
              </Select>
            </Form.Item>
            {config.retrieval?.mode === 'hybrid' && (
              <>
                <Form.Item label={`稠密权重: ${config.retrieval?.hybrid?.dense_weight || 0.7}`}>
                  <Slider min={0} max={1} step={0.05} value={config.retrieval?.hybrid?.dense_weight || 0.7} onChange={v => { setConfig('retrieval.hybrid.dense_weight', v); setConfig('retrieval.hybrid.sparse_weight', +(1 - v).toFixed(2)) }} />
                </Form.Item>
                <Text type="secondary">稀疏权重: {config.retrieval?.hybrid?.sparse_weight || 0.3}</Text>
              </>
            )}
            <Form.Item label="Top-K">
              <InputNumber min={1} max={100} value={config.retrieval?.top_k || 20} onChange={v => setConfig('retrieval.top_k', v)} />
            </Form.Item>
          </Form>
        </Card>
      )
    },
    {
      key: 'reranker',
      label: '📊 重排序',
      children: (
        <Card size="small">
          <Form layout="vertical">
            <Form.Item label="启用重排序">
              <Switch checked={config.reranker?.enabled} onChange={v => setConfig('reranker.enabled', v)} />
            </Form.Item>
            {config.reranker?.enabled && (
              <>
                <Form.Item label="Reranker 模型">
                  <Select value={config.reranker?.provider} onChange={v => setConfig('reranker.provider', v)}>
                    <Option value="bge">BGE Reranker</Option>
                    <Option value="cohere">Cohere Rerank</Option>
                    <Option value="none">无 (关闭)</Option>
                  </Select>
                </Form.Item>
                <Form.Item label="Rerank Top-K">
                  <InputNumber min={1} max={20} value={config.reranker?.top_k || 5} onChange={v => setConfig('reranker.top_k', v)} />
                </Form.Item>
              </>
            )}
          </Form>
        </Card>
      )
    },
    {
      key: 'chunking',
      label: '✂️ 分块',
      children: (
        <Card size="small">
          <Form layout="vertical">
            <Form.Item label="分块策略">
              <Select value={config.chunking?.strategy} onChange={v => setConfig('chunking.strategy', v)}>
                <Option value="fixed">固定长度</Option>
                <Option value="recursive">递归分割</Option>
                <Option value="semantic">语义分块</Option>
              </Select>
            </Form.Item>
            {config.chunking?.strategy === 'fixed' && (
              <Row gutter={16}>
                <Col span={12}><Form.Item label="块大小"><InputNumber min={100} max={4000} value={config.chunking?.fixed?.chunk_size || 512} onChange={v => setConfig('chunking.fixed.chunk_size', v)} style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={12}><Form.Item label="重叠大小"><InputNumber min={0} max={500} value={config.chunking?.fixed?.chunk_overlap || 50} onChange={v => setConfig('chunking.fixed.chunk_overlap', v)} style={{ width: '100%' }} /></Form.Item></Col>
              </Row>
            )}
            {config.chunking?.strategy === 'semantic' && (
              <Row gutter={16}>
                <Col span={8}><Form.Item label="最大块大小"><InputNumber min={200} max={4000} value={config.chunking?.semantic?.max_chunk_size || 1024} onChange={v => setConfig('chunking.semantic.max_chunk_size', v)} style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={8}><Form.Item label="最小块大小"><InputNumber min={50} max={500} value={config.chunking?.semantic?.min_chunk_size || 100} onChange={v => setConfig('chunking.semantic.min_chunk_size', v)} style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={8}><Form.Item label={`相似度阈值: ${config.chunking?.semantic?.similarity_threshold || 0.75}`}><Slider min={0} max={1} step={0.05} value={config.chunking?.semantic?.similarity_threshold || 0.75} onChange={v => setConfig('chunking.semantic.similarity_threshold', v)} /></Form.Item></Col>
              </Row>
            )}
          </Form>
        </Card>
      )
    },
    {
      key: 'agent',
      label: '🧠 Agent',
      children: (
        <Card size="small">
          <Form layout="vertical">
            <Form.Item label="Agent 策略">
              <Select value={config.agent?.strategy} onChange={v => setConfig('agent.strategy', v)}>
                <Option value="react">ReAct (推理+行动)</Option>
                <Option value="planner">Plan-and-Solve (规划式)</Option>
                <Option value="function_calling">Function Calling (函数调用)</Option>
              </Select>
            </Form.Item>
            <Form.Item label="最大迭代次数">
              <InputNumber min={1} max={20} value={config.agent?.max_iterations || 10} onChange={v => setConfig('agent.max_iterations', v)} />
            </Form.Item>
          </Form>
        </Card>
      )
    },
    {
      key: 'multimodal',
      label: '🖼️ 多模态',
      children: (
        <Card size="small">
          <Form layout="vertical">
            <Form.Item label="启用 OCR">
              <Switch checked={config.multimodal?.ocr?.enabled} onChange={v => setConfig('multimodal.ocr.enabled', v)} />
            </Form.Item>
            {config.multimodal?.ocr?.enabled && (
              <Form.Item label="OCR 引擎">
                <Select value={config.multimodal?.ocr?.provider} onChange={v => setConfig('multimodal.ocr.provider', v)}>
                  <Option value="tesseract">Tesseract</Option>
                  <Option value="paddle">PaddleOCR</Option>
                </Select>
              </Form.Item>
            )}
            <Divider />
            <Form.Item label="启用 ASR">
              <Switch checked={config.multimodal?.asr?.enabled} onChange={v => setConfig('multimodal.asr.enabled', v)} />
            </Form.Item>
            {config.multimodal?.asr?.enabled && (
              <Form.Item label="ASR 引擎">
                <Select value={config.multimodal?.asr?.provider} onChange={v => setConfig('multimodal.asr.provider', v)}>
                  <Option value="whisper">Whisper (本地)</Option>
                  <Option value="api">API (远程)</Option>
                </Select>
              </Form.Item>
            )}
          </Form>
        </Card>
      )
    },
  ]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4}>系统配置</Title>
        <div>
          <Button icon={<ReloadOutlined />} onClick={handleReset} style={{ marginRight: 8 }}>重置</Button>
          <Button type="primary" icon={<SaveOutlined />} onClick={handleSave} loading={saving}>保存并生效</Button>
        </div>
      </div>
      <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
        修改配置后点击"保存并生效"，无需重启服务即可实时生效。
      </Text>
      <Tabs items={tabItems} tabPosition="left" />
    </div>
  )
}

export default ConfigPanel