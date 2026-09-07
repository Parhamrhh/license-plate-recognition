// src/pages/AdminPanel.jsx
import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import API_BASE_URL from '../config';

import {
  Layout,
  Menu,
  Table,
  Button,
  Card,
  Modal,
  Form,
  Input,
  Switch,
  Popconfirm,
  message,
  Tabs,
  Select,
  Space
} from 'antd';

import {
  PlusOutlined,
  DeleteOutlined,
  LogoutOutlined
} from '@ant-design/icons';

import { useNavigate } from 'react-router-dom';

import IranianPlate from '../components/IranianPlate';
import SpanishPlate from '../components/SpanishPlate';

const { Header, Content, Sider } = Layout;

const IRAN_LETTERS = [
  'ب', 'د', 'ع', 'ه', 'ح', 'ج', 'ل', 'م',
  'ن', 'پ', 'ق', 'ص', 'س', 'ت', 'ط', 'و',
  'ی', 'ز', 'ش', 'ث', 'ژ', 'الف'
];

const formatPlateRows = (data) => {
  if (!Array.isArray(data)) return [];

  return data.map((item) => ({
    key: `${item.country || 'spain'}-${item.plate}`,
    plate: item.plate,
    country: item.country || 'spain',
    authorized: item.authorized === 'True'
  }));
};

const CameraFeed = ({ frameSrc, detectedPlate }) => {
  return (
    <div style={{ textAlign: 'center' }}>
      {frameSrc && (
        <div style={{ marginBottom: '12px' }}>
          {detectedPlate ? (
            <>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: '12px',
                  flexWrap: 'wrap',
                  fontWeight: 600
                }}
              >
                <span>Detected plate:</span>

                {detectedPlate.country === 'iran' ? (
                  <IranianPlate
                    plate={detectedPlate.plate}
                    parts={detectedPlate.plate_parts}
                  />
                ) : (
                  <SpanishPlate plate={detectedPlate.plate} />
                )}

                {detectedPlate.authorized ? (
                  <span style={{ color: 'green' }}>AUTHORIZED</span>
                ) : (
                  <span style={{ color: 'red' }}>NOT AUTHORIZED</span>
                )}
              </div>

              <div style={{ marginTop: '8px', color: '#555' }}>
                Recognition Confidence:{' '}
                <strong>
                  {(Number(detectedPlate.confidence || 0) * 100).toFixed(1)}%
                </strong>

                {detectedPlate.detection_confidence != null && (
                  <>
                    {' | '}Detection Confidence:{' '}
                    <strong>
                      {(Number(detectedPlate.detection_confidence) * 100).toFixed(1)}%
                    </strong>
                  </>
                )}
              </div>
            </>
          ) : (
            <div style={{ color: '#888' }}>No plate detected</div>
          )}
        </div>
      )}

      {frameSrc ? (
        <img
          src={frameSrc}
          alt="Camera"
          style={{
            width: '100%',
            maxWidth: '640px',
            height: 'auto',
            border: '1px solid #d9d9d9',
            borderRadius: '10px',
            objectFit: 'contain'
          }}
        />
      ) : (
        <p>Waiting for camera...</p>
      )}
    </div>
  );
};

const AdminPanel = () => {
  const navigate = useNavigate();

  const [plates, setPlates] = useState([]);
  const [activeCountry, setActiveCountry] = useState('iran');

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [addForm] = Form.useForm();

  const [frameSrc, setFrameSrc] = useState(null);
  const [detectedPlate, setDetectedPlate] = useState(null);

  const socketRef = useRef(null);

  const [updatingPlates, setUpdatingPlates] = useState(new Set());

  const getRecordKey = (record) => {
    return `${record.country}-${record.plate}`;
  };

  const markUpdating = (record, isUpdating) => {
    const key = getRecordKey(record);

    setUpdatingPlates(prev => {
      const next = new Set(prev);

      if (isUpdating) {
        next.add(key);
      } else {
        next.delete(key);
      }

      return next;
    });
  };

  const loadPlates = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/plates`);

      if (!res.ok) {
        throw new Error('Failed to load plates');
      }

      const data = await res.json();
      setPlates(formatPlateRows(data));
    } catch (err) {
      console.error('Failed to load plates:', err);
      message.error('Could not load plates');
    }
  };

  const openModal = () => {
    addForm.resetFields();

    if (activeCountry === 'iran') {
      addForm.setFieldsValue({
        middle: 'ب'
      });
    }

    setIsModalOpen(true);
  };

  const onAddPlate = async (values) => {
    let newPlate = '';

    if (activeCountry === 'iran') {
      newPlate = `${values.first}${values.middle}${values.serial}${values.region}`;
    } else {
      newPlate = values.plate.trim().toUpperCase();
    }

    try {
      const res = await fetch(`${API_BASE_URL}/plates`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          plate: newPlate,
          country: activeCountry
        })
      });

      const data = await res.json();

      if (!res.ok) {
        message.error(data.error || 'Failed to add plate');
        return;
      }

      message.success(`Added plate "${newPlate}"`);
      setIsModalOpen(false);
      addForm.resetFields();

      await loadPlates();
    } catch (err) {
      console.error(err);
      message.error('Server error');
    }
  };

  const toggleAuthorized = async (record) => {
    markUpdating(record, true);

    try {
      const url =
        `${API_BASE_URL}/plates/${encodeURIComponent(record.plate)}` +
        `?country=${record.country}`;

      const res = await fetch(url, {
        method: 'PATCH'
      });

      if (!res.ok) {
        const data = await res.json();
        message.error(data.error || 'Failed to toggle');
        return;
      }

      await loadPlates();

      message.success(
        `Updated ${record.plate}`
      );
    } catch (err) {
      console.error(err);
      message.error('Server error');
    } finally {
      markUpdating(record, false);
    }
  };

  const deletePlate = async (record) => {
    try {
      const url =
        `${API_BASE_URL}/plates/${encodeURIComponent(record.plate)}` +
        `?country=${record.country}`;

      const res = await fetch(url, {
        method: 'DELETE'
      });

      if (!res.ok) {
        const data = await res.json();
        message.error(data.error || 'Failed to delete');
        return;
      }

      message.success(`Deleted plate "${record.plate}"`);

      await loadPlates();
    } catch (err) {
      console.error(err);
      message.error('Server error');
    }
  };

  useEffect(() => {
    loadPlates();

    const socket = io(API_BASE_URL, {
      transports: ['websocket']
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('socket connected', socket.id);
    });

    socket.on('disconnect', () => {
      console.log('socket disconnected');
    });

    socket.on('frame', (data) => {
      if (data && data.image) {
        setFrameSrc(
          `data:image/jpeg;base64,${data.image}`
        );
      }
    });

    socket.on('plate_detected', (data) => {
      if (!data) return;

      const payload = {
        plate: data.plate || data?.plate_text || '',
        country: data.country || 'spain',
        plate_type: data.plate_type || null,
        plate_parts: data.plate_parts || null,
        confidence:
          data.confidence != null
            ? Number(data.confidence)
            : Number(data.prob || 0),
        detection_confidence:
          data.detection_confidence != null
            ? Number(data.detection_confidence)
            : null,
        authorized: !!data.authorized
      };

      setDetectedPlate(payload);

      setPlates(prev =>
        prev.map(item =>
          item.plate === payload.plate &&
          item.country === payload.country
            ? {
                ...item,
                lastSeenAuthorized: payload.authorized
              }
            : item
        )
      );
    });

    socket.on('plates_list', (data) => {
      try {
        setPlates(formatPlateRows(data));
      } catch (err) {
        console.error('Malformed plates_list', err);
      }
    });

    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, []);

  const handleLogout = () => {
    sessionStorage.clear();
    navigate('/login');
  };

  const filteredPlates = plates.filter(
    item => item.country === activeCountry
  );

  const columns = [
    {
      title: 'Plate',
      dataIndex: 'plate',
      key: 'plate',
      render: (_, record) => (
        record.country === 'iran' ? (
          <IranianPlate plate={record.plate} />
        ) : (
          <SpanishPlate plate={record.plate} />
        )
      )
    },
    {
      title: 'Authorized?',
      dataIndex: 'authorized',
      key: 'authorized',
      render: (auth, record) => {
        const key = getRecordKey(record);

        return (
          <Switch
            checked={auth}
            onChange={() => toggleAuthorized(record)}
            loading={updatingPlates.has(key)}
            disabled={updatingPlates.has(key)}
          />
        );
      }
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, record) => (
        <Popconfirm
          title={`Delete "${record.plate}"?`}
          onConfirm={() => deletePlate(record)}
          okText="Yes"
          cancelText="No"
        >
          <Button danger icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider>
        <div
          style={{
            height: 32,
            margin: 16,
            color: 'white',
            textAlign: 'center',
            fontSize: '1.25rem'
          }}
        >
          Admin Panel
        </div>

        <Menu theme="dark" mode="inline" defaultSelectedKeys={['logout']}>
          <Menu.Item
            key="logout"
            icon={<LogoutOutlined />}
            onClick={handleLogout}
          >
            Logout
          </Menu.Item>
        </Menu>
      </Sider>

      <Layout>
        <Header
          style={{
            background: '#fff',
            padding: '0 16px',
            display: 'flex',
            alignItems: 'center'
          }}
        >
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={openModal}
          >
            Add Plate
          </Button>
        </Header>

        <Content style={{ margin: '16px' }}>
          <Card title="Plates List" bordered={false}>
            <Tabs
              activeKey={activeCountry}
              onChange={setActiveCountry}
              items={[
                {
                  key: 'iran',
                  label: 'Iranian Plates'
                },
                {
                  key: 'spain',
                  label: 'Spanish Plates'
                }
              ]}
            />

            <Table
              dataSource={filteredPlates}
              columns={columns}
              pagination={false}
              rowKey={record => getRecordKey(record)}
            />
          </Card>

          <Card
            title="Camera Feed"
            bordered={false}
            style={{ marginTop: '16px' }}
          >
            <CameraFeed
              frameSrc={frameSrc}
              detectedPlate={detectedPlate}
            />
          </Card>
        </Content>
      </Layout>

      <Modal
        title={
          activeCountry === 'iran'
            ? 'Add Iranian Plate'
            : 'Add Spanish Plate'
        }
        open={isModalOpen}
        onCancel={() => setIsModalOpen(false)}
        footer={null}
      >
        <Form
          form={addForm}
          layout="vertical"
          onFinish={onAddPlate}
        >
          {activeCountry === 'iran' ? (
            <>
              <div
                style={{
                  marginBottom: '16px',
                  padding: '12px',
                  background: '#fafafa',
                  borderRadius: '8px'
                }}
              >
                <Space align="start" wrap>
                  <Form.Item
                    label="First"
                    name="first"
                    rules={[
                      {
                        required: true,
                        message: 'Required'
                      },
                      {
                        pattern: /^[0-9۰-۹٠-٩]{2}$/,
                        message: '2 digits'
                      }
                    ]}
                  >
                    <Input
                      placeholder="12"
                      maxLength={2}
                      style={{ width: 65 }}
                      inputMode="numeric"
                    />
                  </Form.Item>

                  <Form.Item
                    label="Letter"
                    name="middle"
                    rules={[
                      {
                        required: true,
                        message: 'Required'
                      }
                    ]}
                  >
                    <Select
                      style={{ width: 110 }}
                      options={IRAN_LETTERS.map(letter => ({
                        value: letter,
                        label:
                          letter === 'ژ'
                            ? '♿ معلولین'
                            : letter
                      }))}
                    />
                  </Form.Item>

                  <Form.Item
                    label="Serial"
                    name="serial"
                    rules={[
                      {
                        required: true,
                        message: 'Required'
                      },
                      {
                        pattern: /^[0-9۰-۹٠-٩]{3}$/,
                        message: '3 digits'
                      }
                    ]}
                  >
                    <Input
                      placeholder="365"
                      maxLength={3}
                      style={{ width: 75 }}
                      inputMode="numeric"
                    />
                  </Form.Item>

                  <div
                    style={{
                      fontSize: '28px',
                      marginTop: '29px'
                    }}
                  >
                    |
                  </div>

                  <Form.Item
                    label="Region"
                    name="region"
                    rules={[
                      {
                        required: true,
                        message: 'Required'
                      },
                      {
                        pattern: /^[0-9۰-۹٠-٩]{2}$/,
                        message: '2 digits'
                      }
                    ]}
                  >
                    <Input
                      placeholder="11"
                      maxLength={2}
                      style={{ width: 65 }}
                      inputMode="numeric"
                    />
                  </Form.Item>
                </Space>
              </div>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  block
                >
                  Add Iranian Plate
                </Button>
              </Form.Item>
            </>
          ) : (
            <>
              <Form.Item
                label="Plate Number"
                name="plate"
                rules={[
                  {
                    required: true,
                    message: 'Please enter a plate number'
                  },
                  {
                    pattern: /^\d{4}[A-Z]{3}$/,
                    message: 'Format must be 1234ABC'
                  }
                ]}
              >
                <Input
                  placeholder="e.g. 4130DVM"
                  maxLength={7}
                  onChange={event => {
                    addForm.setFieldsValue({
                      plate: event.target.value.toUpperCase()
                    });
                  }}
                />
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  block
                >
                  Add Spanish Plate
                </Button>
              </Form.Item>
            </>
          )}
        </Form>
      </Modal>
    </Layout>
  );
};

export default AdminPanel;