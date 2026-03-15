import React, { useState, useEffect } from 'react'
import { Card, Select, Button, Table, Typography, Row, Col, Statistic, Spin, InputNumber, Switch, message } from 'antd'
import { BarChartOutlined, PlayCircleOutlined } from '@ant-design/icons'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { listDatasets, runEvaluation } from '../api/client'
import useConfigStore from '../stores/configStore'

const { Title, Text } = Typography
const { Option } = Select

const EvaluationPanel = () => {
  const { sessionId } = useConfigStore()
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [topK, setTopK] = useState(10)
  const [useReranker, setUseReranker] = useState(true)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    listDatasets(sessionId).then(res => {
      setDatasets(res.data)
      if (res.data.length > 0) setSelectedDataset(res.data[0].name)
    }).catch(() => {})
  }, [])

  const handleRun = async () => {
    if (!selectedDataset) { message.warning('请选择数据集'); return }
    setLoading(true)
    try {
      const res = await runEvaluation({
        session_id: sessionId,
        dataset_name: selectedDataset,
        top_k: topK,
        use_reranker: useReranker,
      })
      setResults(res.data)
      message.success('评测完成!')
    } catch (err) {
      message.error('评测失败: ' + (err.response?.data?.detail || err.message))
    }
    setLoading(false)
  }

  const chartData = results ? Object.entries(results.metrics).map(([k, v]) => ({ name: k.toUpperCase(), score: v || 0 })) : []

  return (
    <div>
      <Title level={4}>RAG 效果评测</Title>

      <Card style={{ marginBottom: 24 }}>
        <Row gutter={16} align="middle">
          <Col span={6}>
            <Text strong>数据集:</Text>
            <Select value={selectedDataset} onChange={setSelectedDataset} style={{ width: '100%', marginTop: 4 }}>
              {datasets.map(ds => <Option key={ds.name} value={ds.name}>{ds.name} ({ds.num_queries} queries)</Option>)}
            </Select>
          </Col>
          <Col span={4}>
            <Text strong>Top-K:</Text>
            <InputNumber min={1} max={100} value={topK} onChange={setTopK} style={{ width: '100%', marginTop: 4 }} />
          </Col>
          <Col span={4}>
            <Text strong>启用重排序:</Text><br />
            <Switch checked={useReranker} onChange={setUseReranker} style={{ marginTop: 8 }} />
          </Col>
          <Col span={4}>
            <Button type="primary" icon={<PlayCircleOutlined />} onClick={handleRun} loading={loading} style={{ marginTop: 20 }}>
              开始评测
            </Button>
          </Col>
        </Row>
      </Card>

      {loading && <Spin size="large" style={{ display: 'block', textAlign: 'center', margin: 48 }} />}

      {results && !loading && (
        <>
          <Row gutter={16} style={{ marginBottom: 24 }}>
            {Object.entries(results.metrics).map(([k, v]) => (
              <Col span={4} key={k}>
                <Card><Statistic title={k.toUpperCase()} value={v !== null ? (v * 100).toFixed(1) + '%' : 'N/A'} /></Card>
              </Col>
            ))}
          </Row>

          <Card title="指标可视化" style={{ marginBottom: 24 }}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={v => (v * 100).toFixed(1) + '%'} />
                <Bar dataKey="score" fill="#1677ff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="查询详情">
            <Table
              dataSource={results.query_details?.map((q, i) => ({ ...q, key: i }))}
              columns={[
                { title: '查询', dataIndex: 'query', key: 'query' },
                { title: '检索数量', dataIndex: 'num_retrieved', key: 'num_retrieved' },
                { title: '相关文档', dataIndex: 'relevant', key: 'relevant', render: r => r?.join(', ') || '-' },
              ]}
              size="small"
              pagination={false}
            />
          </Card>
        </>
      )}
    </div>
  )
}

export default EvaluationPanel