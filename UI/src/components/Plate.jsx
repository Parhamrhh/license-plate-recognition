// src/components/Plate.jsx
import React from 'react';

/**
 * Renders a license‐plate‐style box with modern styling.
 * @param {{ text: string, authorized: boolean, onClick: Function }} props
 */
const Plate = ({ text, authorized = false, onClick = null }) => {
  // Modern color palette
  const borderColor = authorized ? '#38bdf8' : '#fb7185';
  const textColor = authorized ? '#0369a1' : '#be123c';
  const bgColor = authorized ? '#e0f2fe' : '#ffe4e6';
  const indicatorColor = '#38bdf8';
  
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 130,
        height: 60,
        backgroundColor: bgColor,
        border: `2px solid ${borderColor}`,
        borderRadius: 8,
        boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
        fontFamily: '"Roboto Mono", monospace',
        fontSize: 22,
        fontWeight: '600',
        letterSpacing: 2,
        color: textColor,
        transition: 'all 0.3s ease',
        cursor: onClick ? 'pointer' : 'default',
        position: 'relative',
        overflow: 'hidden',
        userSelect: 'none',
        transform: 'translateY(0)',
      }}
      onClick={onClick}
      title={authorized ? "Authorized plate" : "Unauthorized plate"}
      onMouseOver={(e) => {
        if (onClick) {
          e.currentTarget.style.transform = 'translateY(-2px)';
          e.currentTarget.style.boxShadow = '0 6px 16px rgba(0,0,0,0.12)';
        }
      }}
      onMouseOut={(e) => {
        if (onClick) {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.08)';
        }
      }}
    >
      {text}
      {authorized && (
        <div style={{
          position: 'absolute',
          top: 4,
          right: 4,
          width: 5,
          height: 5,
          borderRadius: '50%',
          backgroundColor: indicatorColor,
          boxShadow: '0 0 4px rgba(56, 189, 248, 0.6)',
        }} />
      )}
    </div>
  );
};

export default Plate;