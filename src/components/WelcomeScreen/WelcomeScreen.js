import React, { useState } from 'react';

export default function WelcomeScreen({ onStart }) {
  const container = {
    height: '100vh',
    width: '100vw',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    background: '#24292e',
  };
  const title = {
    color: '#fafafa',
    fontSize: '3rem',
    marginBottom: '2rem',
  };
  const buttonBase = {
    background: '#2ea44f',
    color: 'white',
    border: 'none',
    padding: '1rem 2rem',
    fontSize: '1.25rem',
    borderRadius: '6px',
    cursor: 'pointer',
    boxShadow: '0 4px 6px rgba(0,0,0,0.2)',
    transition: 'background 0.2s',
  };
  const [hover, setHover] = useState(false);

  return (
    <div style={container}>
      <h1 style={title}>Welcome to HT Coding</h1>
      <button
        style={{
          ...buttonBase,
          background: hover ? '#2c974b' : buttonBase.background
        }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        onClick={onStart}
      >
        Start Coding
      </button>
    </div>
  );
}
