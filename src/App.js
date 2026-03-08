import React, { useState, useEffect } from 'react';
import WelcomeScreen from './components/WelcomeScreen/WelcomeScreen';
import EditorLayout from './components/EditorLayout/EditorLayout';
import './App.css';

export default function App() {
  const [started, setStarted] = useState(false);

  // Suppress the ResizeObserver loop warnings
  useEffect(() => {
    const loopErrMsg = 'ResizeObserver loop completed with undelivered notifications.';
    const handler = (e) => {
      if (e.message === loopErrMsg) {
        e.stopImmediatePropagation();
      }
    };
    window.addEventListener('error', handler);
    return () => window.removeEventListener('error', handler);
  }, []);

  return started
    ? <EditorLayout />
    : <WelcomeScreen onStart={() => setStarted(true)} />;
}
