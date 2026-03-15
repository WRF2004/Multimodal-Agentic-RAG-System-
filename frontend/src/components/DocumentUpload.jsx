import React, { useState, useEffect } from 'react'
import { Upload, Card, Typography, Tag, Table, Statistic, Row, Col, message } from 'antd'
import { InboxOutlined, FileTextOutlined } from '@ant-design/icons'
import { uploadDocument, getDocStats } from '../api/client'
import useConfigStore from '../stores/configStore'

const { Dragger } = Upload
const { Title, Text } = Typography

const DocumentUpload = () => {
  const { sessionId } = useConfigStore()
  const [uploadHistory, setUploadHistory] = useState([])
  const [stats, setStats] = useState({ count: 0 })

  useEffect(() => {
    refreshStats()
  }, [])

  const refreshStats = async () => {
    try {
      const res = await getDocStats(sessionId)
      setStats(res.data)
    } catch (e) { /* ignore */ }
  }

  const handleUpload = async (options) => {
    const { file, onSuccess, onError } = options
    try {
      const res = await uploadDocument(file, sessionId)
      const record = { key: res.data.id, filename: res.data.filename, status: res.data.status, time: new Date().toLocaleTimeString() }
      setUploadHistory(prev => [record, ...prev])
      onSuccess(res.data)
      message.success(`${file.name} 上传成功，正在处理...`)
      setTimeout(refreshStats, 3000)
    } catch (err) {
      onError(err)
      message.error(`上传失败: ${err.response?.data?.detail || err.message}`)
    }
  }

  const columns = [
    { title: '文件名', dataIndex: 'filename', key: 'filename' },
    { title: '状态', dataIndex: 'status', key: 'status', render: s => <Tag color={s === 'processing' ? 'blue' : s === 'completed' ? 'green' : 'default'}>{s}</Tag> },
    { title: '时间', dataIndex: 'time', key: 'time' },
  ]

  return (
    <div>
      <Title level={4}>文档管理</Title>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card><Statistic title="知识库文档数" value={stats.count || 0} prefix={<FileTextOutlined />} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="已上传文件" value={uploadHistory.length} /></Card>
        </Col>
        <Col span={8}>
          <Card><Statistic title="集合名称" value={stats.name || 'documents'} valueStyle={{ fontSize: 16 }} /></Card>
        </Col>
      </Row>

      <Card style={{ marginBottom: 24 }}>
        <Dragger customRequest={handleUpload} multiple accept=".pdf,.txt,.md,.docx,.png,.jpg,.jpeg,.mp3,.wav">
          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
          <p className="ant-upload-text">点击或拖拽文件上传</p>
          <p className="ant-upload-hint">支持 PDF、TXT、MD、DOCX、图片(OCR)、音频(ASR)</p>
        </Dragger>
      </Card>

      {uploadHistory.length > 0 && (
        <Card title="上传历史">
          <Table dataSource={uploadHistory} columns={columns} size="small" pagination={false} />
        </Card>
      )}
    </div>
  )
}

export default DocumentUpload