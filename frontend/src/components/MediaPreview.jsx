import React from 'react'
import { Image, Card, Typography } from 'antd'
import { FileImageOutlined, SoundOutlined } from '@ant-design/icons'

const { Text } = Typography

const MediaPreview = ({ source }) => {
  if (!source) return null

  const { file_type, source: path } = source

  if (['image', 'png', 'jpg', 'jpeg'].includes(file_type)) {
    return (
      <Card size="small" style={{ marginTop: 8 }}>
        <FileImageOutlined /> <Text type="secondary">图片来源</Text>
        <Image src={`/api/documents/preview/${encodeURIComponent(path)}`} style={{ maxWidth: 300, marginTop: 8 }} fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..." />
      </Card>
    )
  }

  if (['audio', 'mp3', 'wav'].includes(file_type)) {
    return (
      <Card size="small" style={{ marginTop: 8 }}>
        <SoundOutlined /> <Text type="secondary">音频来源</Text>
        <audio controls src={`/api/documents/preview/${encodeURIComponent(path)}`} style={{ width: '100%', marginTop: 8 }} />
      </Card>
    )
  }

  return null
}

export default MediaPreview