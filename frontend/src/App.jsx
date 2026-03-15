import React, { useState } from 'react'
import { Layout, Menu, Typography } from 'antd'
import {
  MessageOutlined, SettingOutlined, CloudUploadOutlined,
  BarChartOutlined, RobotOutlined
} from '@ant-design/icons'
import ChatPanel from './components/ChatPanel'
import ConfigPanel from './components/ConfigPanel'
import DocumentUpload from './components/DocumentUpload'
import EvaluationPanel from './components/EvaluationPanel'

const { Header, Sider, Content } = Layout
const { Title } = Typography

const App = () => {
  const [activeTab, setActiveTab] = useState('chat')

  const menuItems = [
    { key: 'chat', icon: <MessageOutlined />, label: '对话' },
    { key: 'config', icon: <SettingOutlined />, label: '配置' },
    { key: 'documents', icon: <CloudUploadOutlined />, label: '文档' },
    { key: 'evaluation', icon: <BarChartOutlined />, label: '评测' },
  ]

  const renderContent = () => {
    switch (activeTab) {
      case 'chat': return <ChatPanel />
      case 'config': return <ConfigPanel />
      case 'documents': return <DocumentUpload />
      case 'evaluation': return <EvaluationPanel />
      default: return <ChatPanel />
    }
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', background: '#001529', padding: '0 24px' }}>
        <RobotOutlined style={{ fontSize: 24, color: '#1677ff', marginRight: 12 }} />
        <Title level={4} style={{ color: 'white', margin: 0 }}>Agentic RAG System</Title>
      </Header>
      <Layout>
        <Sider width={200} theme="light" collapsible>
          <Menu
            mode="inline"
            selectedKeys={[activeTab]}
            items={menuItems}
            onClick={({ key }) => setActiveTab(key)}
            style={{ height: '100%' }}
          />
        </Sider>
        <Content style={{ padding: 24, background: '#f5f5f5', overflow: 'auto' }}>
          {renderContent()}
        </Content>
      </Layout>
    </Layout>
  )
}

export default App