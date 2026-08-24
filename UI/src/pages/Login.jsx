import React, { useState } from 'react';
import { Form, Input, Button, Card, message } from 'antd';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const onFinish = ({ username, password }) => {
    setLoading(true);

    // Simulate an API call delay
    setTimeout(() => {
      if (username === 'admin') {
        // Successful admin login
        sessionStorage.setItem('token', 'dummy-admin-token');
        sessionStorage.setItem('role', 'admin');
        message.success('Logged in as admin');
        navigate('/admin');
      } else {
        // Any non-admin user is not allowed—redirect back and show error
        sessionStorage.clear();
        message.error('Only admin can log in');
        navigate('/login');
      }
      setLoading(false);
    }, 800);
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh'
      }}
    >
      <Card title="Admin Login" style={{ width: 320 }}>
        <Form
          name="login_form"
          layout="vertical"
          initialValues={{ username: 'admin', password: '' }}
          onFinish={onFinish}
        >
          <Form.Item
            label="Username"
            name="username"
            rules={[{ required: true, message: 'Please enter your username' }]}
          >
            <Input placeholder="e.g. admin" />
          </Form.Item>

          <Form.Item
            label="Password"
            name="password"
            rules={[{ required: true, message: 'Please enter your password' }]}
          >
            <Input.Password placeholder="any password" />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" block loading={loading}>
              Log In
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default Login;
