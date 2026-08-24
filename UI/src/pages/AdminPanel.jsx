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
  message
} from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

// Plate component (you already have this)
import Plate from '../components/Plate';

const { Header, Content, Sider } = Layout;

const CameraFeed = ({ frameSrc, detectedPlate }) => {
  return (
    <div style={{ textAlign: 'center' }}>
      {frameSrc && (
        <div style={{ marginBottom: '8px' }}>
          {detectedPlate ? (
            <div style={{ fontWeight: 600 }}>
              Detected plate: {detectedPlate.plate} &nbsp; — &nbsp;
              {detectedPlate.authorized ? (
                <span style={{ color: 'green' }}>AUTHORIZED</span>
              ) : (
                <span style={{ color: 'red' }}>NOT AUTHORIZED</span>
              )}
              &nbsp; ({Number(detectedPlate.confidence).toFixed(2)})
            </div>
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
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [addForm] = Form.useForm();

  // realtime states
  const [frameSrc, setFrameSrc] = useState(null);
  const [detectedPlate, setDetectedPlate] = useState(null);

  // keep socket ref so we can reuse/cleanup
  const socketRef = useRef(null);

  // track rows being updated to avoid optimistic mismatches
  const [updatingPlates, setUpdatingPlates] = useState(new Set());
  const markUpdating = (plate, isUpdating) => {
    setUpdatingPlates(prev => {
      const next = new Set(prev);
      if (isUpdating) next.add(plate);
      else next.delete(plate);
      return next;
    });
  };

  const openModal = () => {
    addForm.resetFields();
    setIsModalOpen(true);
  };

  const onAddPlate = async (values) => {
    const newPlate = values.plate.trim();

    try {
      const res = await fetch(`${API_BASE_URL}/plates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plate: newPlate }),
      });

      if (!res.ok) {
        const err = await res.json();
        message.error(err.error || 'Failed to add plate');
        return;
      }

      // optimistic update (server will also broadcast plates_list)
      setPlates(prev => {
        if (prev.some(p => p.plate === newPlate)) return prev;
        return [...prev, { key: newPlate, plate: newPlate, authorized: false }];
      });

      message.success(`Added plate "${newPlate}"`);
      setIsModalOpen(false);
    } catch (err) {
      console.error(err);
      message.error('Server error');
    }
  };

  const toggleAuthorized = async (record) => {
    // show loading on the switch to prevent rapid flips and stale UI
    markUpdating(record.plate, true);
    try {
      const res = await fetch(`${API_BASE_URL}/plates/${record.plate}`, {
        method: 'PATCH',
      });

      if (!res.ok) {
        message.error('Failed to toggle');
        return;
      }

      // Hard refresh this list from server to avoid optimistic mismatches
      const listRes = await fetch(`${API_BASE_URL}/plates`);
      if (listRes.ok) {
        const data = await listRes.json();
        const formatted = data.map((item) => ({
          key: item.plate,
          plate: item.plate,
          authorized: item.authorized === 'True',
        }));
        setPlates(formatted);
        const current = formatted.find(p => p.plate === record.plate);
        if (current) {
          message.success(`${record.plate} is now ${current.authorized ? 'Authorized' : 'Denied'}`);
        } else {
          message.success(`Updated ${record.plate}`);
        }
      } else {
        // fallback: rely on socket 'plates_list' to arrive shortly
        message.success(`Toggled ${record.plate}`);
      }
    } catch (err) {
      console.error(err);
      message.error('Server error');
    } finally {
      markUpdating(record.plate, false);
    }
  };

  useEffect(() => {
    // initial fetch (fallback if socket doesn't immediately send plates_list)
    fetch(`${API_BASE_URL}/plates`)
      .then(res => res.json())
      .then(data => {
        const formatted = data.map((item) => ({
          key: item.plate,
          plate: item.plate,
          authorized: item.authorized === 'True',
        }));
        setPlates(formatted);
      })
      .catch(err => {
        console.error('Failed to load plates:', err);
        message.error('Could not load plates');
      });

    // create socket and listeners
    const socket = io(API_BASE_URL, { transports: ['websocket'] });
    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('socket connected', socket.id);
    });

    socket.on('disconnect', () => {
      console.log('socket disconnected');
    });

    // frame event: server sends { image: "<base64 string>" }
    socket.on('frame', (data) => {
      if (data && data.image) {
        setFrameSrc(`data:image/jpeg;base64,${data.image}`);
      }
    });

    // plate_detected: { plate, confidence, authorized }
    socket.on('plate_detected', (d) => {
      if (!d) return;
      // normalize incoming payload shape safety
      const payload = {
        plate: d.plate || d?.plate_text || '',
        confidence: d.confidence != null ? d.confidence : (d.prob || 0),
        authorized: !!d.authorized
      };
      setDetectedPlate(payload);

      // optional: highlight or update row state to reflect recent detection
      setPlates(prev => prev.map(p => p.plate === payload.plate ? { ...p, lastSeenAuthorized: payload.authorized } : p));
    });

    // receive full plates list broadcasted from backend
    socket.on('plates_list', (data) => {
      if (!Array.isArray(data)) return;
      try {
        const formatted = data.map((item) => ({
          key: item.plate,
          plate: item.plate,
          authorized: item.authorized === 'True'
        }));
        setPlates(formatted);
      } catch (err) {
        console.error('Malformed plates_list', err);
      }
    });

    return () => {
      if (socket) socket.disconnect();
    };
  }, []);

  const deletePlate = async (record) => {
    try {
      const res = await fetch(`${API_BASE_URL}/plates/${record.plate}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        // optimistic removal (server will broadcast plates_list too)
        setPlates(prev => prev.filter(p => p.plate !== record.plate));
        message.success(`Deleted plate "${record.plate}"`);
      } else {
        message.error('Failed to delete');
      }
    } catch (err) {
      console.error(err);
      message.error('Server error');
    }
  };

  const handleLogout = () => {
    sessionStorage.clear();
    navigate('/login');
  };

  const columns = [
    {
      title: 'Plate',
      dataIndex: 'plate',
      key: 'plate',
      render: (_, record) => (
        <Plate text={record.plate} authorized={record.authorized} />
      )
    },
    {
      title: 'Authorized?',
      dataIndex: 'authorized',
      key: 'authorized',
      render: (auth, record) => (
        <Switch
          checked={auth}
          onChange={() => toggleAuthorized(record)}
          loading={updatingPlates.has(record.plate)}
          disabled={updatingPlates.has(record.plate)}
        />
      )
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
            <Table
              dataSource={plates}
              columns={columns}
              pagination={false}
              rowKey="plate"
            />
          </Card>

          <Card title="Camera Feed" bordered={false} style={{ marginTop: '16px' }}>
            <CameraFeed frameSrc={frameSrc} detectedPlate={detectedPlate} />
          </Card>
        </Content>
      </Layout>

      <Modal
        title="Add a New Plate"
        open={isModalOpen}
        onCancel={() => setIsModalOpen(false)}
        footer={null}
      >
        <Form form={addForm} layout="vertical" onFinish={onAddPlate}>
          <Form.Item
            label="Plate Number"
            name="plate"
            rules={[
              { required: true, message: 'Please enter a plate number' },
              {
                pattern: /^[A-Z0-9]{1,8}$/,
                message:
                  'Use only uppercase letters/numbers (max 8 chars)'
              }
            ]}
          >
            <Input placeholder="e.g. GHI789" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" block>
              Add Plate
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </Layout>
  );
};

export default AdminPanel;
