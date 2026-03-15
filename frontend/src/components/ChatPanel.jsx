import React, { useState, useRef, useEffect } from 'react'
import { Input, Button, Card, List, Tag, Space, Spin, Typography, Collapse } from 'antd'
import { SendOutlined, ClearOutlined, ToolOutlined } from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import useConfigStore from '../stores/configStore'
import { createChatWS } from '../api/client'

const { TextArea } = Input
const { Text } = Typography

const ChatPanel = () => {
  const { sessionId, config, messages, addMessage, setLoading, isLoading, clearMessages } = useConfigStore()
  const [input, setInput] = useState('')
  const [agentSteps, setAgentSteps] = useState([])
  const messagesEndRef = useRef(null)
  const wsRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, agentSteps])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return
    const userMsg = input.trim()
    setInput('')
    addMessage({ role: 'user', content: userMsg })
    setLoading(true)
    setAgentSteps([])

    try {
      const ws = createChatWS(sessionId)
      wsRef.current = ws

      ws.onopen = () => {
        ws.send(JSON.stringify({
          message: userMsg,
          config_overrides: config,
        }))
      }

      let fullAnswer = ''
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        switch (data.type) {
          case 'thinking':
            setAgentSteps(prev => [...prev, { type: 'thinking', content: data.content }])
            break
          case 'action':
            setAgentSteps(prev => [...prev, {
              type: 'action', tool: data.tool, input: data.input, reasoning: data.reasoning
            }])
            break
          case 'observation':
            setAgentSteps(prev => [...prev, { type: 'observation', content: data.content }])
            break
          case 'plan':
            setAgentSteps(prev => [...prev, { type: 'plan', content: data.content }])
            break
          case 'answer':
            fullAnswer = data.content
            addMessage({ role: 'assistant', content: data.content })
            break
          case 'token':
            fullAnswer += data.content
            break
          case 'done':
            if (fullAnswer && !messages.find(m => m.content === fullAnswer)) {
              addMessage({ role: 'assistant', content: fullAnswer })
            }
            setLoading(false)
            ws.close()
            break
        }
      }

      ws.onerror = () => {
        addMessage({ role: 'assistant', content: '连接出错，请重试。' })
        setLoading(false)
      }

      ws.onclose = () => setLoading(false)

    } catch (err) {
      addMessage({ role: 'assistant', content: `错误: ${err.message}` })
      setLoading(false)
    }
  }

  const renderStep = (step, idx) => {
    if (step.type === 'thinking') {
      return <Tag color="blue" key={idx}>{step.content}</Tag>
    } else if (step.type === 'action') {
      return (
        <Card size="small" key={idx} style={{ marginBottom: 4, background: '#f6ffed' }}>
          <Space><ToolOutlined /><Text strong>工具: {step.tool}</Text></Space>
          <br /><Text type="secondary">{step.reasoning}</Text>
        </Card>
      )
    } else if (step.type === 'observation') {
      return (
        <Card size="small" key={idx} style={{ marginBottom: 4, background: '#fff7e6' }}>
          <Text>{step.content?.substring(0, 300)}{step.content?.length > 300 ? '...' : ''}</Text>
        </Card>
      )
    }
    return null
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)' }}>
      {/* Messages Area */}
      <div style={{ flex: 1, overflow: 'auto', padding: '0 16px' }}>
        <List
          dataSource={messages}
          renderItem={(msg, idx) => (
            <List.Item style={{ border: 'none', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
              <Card
                size="small"
                style={{
                  maxWidth: '75%',
                  background: msg.role === 'user' ? '#1677ff' : '#fff',
                  color: msg.role === 'user' ? '#fff' : '#000',
                  borderRadius: 12,
                }}
                styles={{ body: { padding: '8px 16px' } }}
              >
                {msg.role === 'user' ? (
                  <Text style={{ color: '#fff' }}>{msg.content}</Text>
                ) : (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                )}
              </Card>
            </List.Item>
          )}
        />

        {/* Agent Steps */}
        {agentSteps.length > 0 && (
          <Collapse
            size="small"
            items={[{
              key: '1',
              label: `Agent 推理过程 (${agentSteps.length} 步)`,
              children: agentSteps.map(renderStep),
            }]}
            style={{ marginBottom: 8 }}
          />
        )}

        {isLoading && <Spin tip="思考中..." style={{ display: 'block', textAlign: 'center', margin: 16 }} />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{ padding: 16, borderTop: '1px solid #f0f0f0', background: '#fff' }}>
        <Space.Compact style={{ width: '100%' }}>
          <TextArea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); handleSend() } }}
            placeholder="输入你的问题... (Shift+Enter 换行)"
            autoSize={{ minRows: 1, maxRows: 4 }}
            style={{ borderRadius: '8px 0 0 8px' }}
          />
          <Button type="primary" icon={<SendOutlined />} onClick={handleSend} loading={isLoading} style={{ height: 'auto' }}>
            发送
          </Button>
          <Button icon={<ClearOutlined />} onClick={clearMessages} style={{ height: 'auto' }}>
            清空
          </Button>
        </Space.Compact>
      </div>
    </div>
  )
}

export default ChatPanel