import React from 'react';
import './plateStyles.css';

function parseIranianPlate(plate) {
  if (!plate || plate.length < 8) return null;

  return {
    first: plate.slice(0, 2),
    middle: plate.slice(2, -5),
    serial: plate.slice(-5, -2),
    region: plate.slice(-2),
  };
}

function getPlateClass(middle) {
  if (middle === 'ت' || middle === 'ع') return 'iran-plate-yellow';
  if (middle === 'الف') return 'iran-plate-red';
  if (middle === 'پ') return 'iran-plate-green';
  return 'iran-plate-white';
}

export default function IranianPlate({ plate, parts }) {
  const data = parts || parseIranianPlate(plate);

  if (!data) return <span>{plate}</span>;

  const middleDisplay = data.middle === 'ژ' ? '♿' : data.middle;

  return (
    <div className={`iran-plate ${getPlateClass(data.middle)}`} dir="ltr">
      <div className="iran-blue-band">
        <span>IR</span>
      </div>

      <span className="iran-plate-number">{data.first}</span>
      <span className="iran-plate-letter" dir="rtl">{middleDisplay}</span>
      <span className="iran-plate-number">{data.serial}</span>

      <span className="iran-region">
        <small>ایران</small>
        <strong>{data.region}</strong>
      </span>
    </div>
  );
}