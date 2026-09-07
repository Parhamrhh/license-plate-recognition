import React from 'react';
import './plateStyles.css';

export default function SpanishPlate({ plate }) {
  return (
    <div className="spain-plate" dir="ltr">
      <div className="spain-blue-band">EU</div>
      <span>{plate}</span>
    </div>
  );
}