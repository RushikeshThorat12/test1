// src/components/EditorLayout/EditorLayout.js
import React, { useState, useRef, useEffect } from 'react';
import Split from 'react-split';
import Editor from '@monaco-editor/react';

const QUESTIONS = [
  { id: 1, title: '1. Two Sum' },
  { id: 2, title: '2. Reverse String' },
  { id: 3, title: '3. Palindrome Number' },
  { id: 4, title: '4. Merge Sorted Array' },
  { id: 5, title: '5. Valid Parentheses' },
  { id: 6, title: '6. Best Time to Buy/Sell Stocks' },
  { id: 7, title: '7. Maximum Subarray' },
  { id: 8, title: '8. Climbing Stairs' },
  { id: 9, title: '9. Symmetric Tree' },
  { id: 10, title: '10. Binary Tree Inorder Traversal' },
];

export default function EditorLayout() {
  const [selected, setSelected] = useState(QUESTIONS[0]);
  const [bottomTab, setBottomTab] = useState('results');
  const rootRef = useRef(null);

  // 1) Inject gutter styles into the subtree before fullscreen,
  //    then request fullscreen on mount
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;

    // Create and inject <style> into the element that will be fullscreened
    const style = document.createElement('style');
    style.textContent = `
      /* remove default black gutters */
      .gutter { background-color: transparent !important; }

      /* horizontal gutter */
      .gutter.gutter-horizontal {
        background-color: var(--color-splitter-bg, #f0f0f0) !important;
        width: 10px !important;
        cursor: col-resize !important;
        border-left: 1px solid var(--color-splitter-border, #d7e1e2) !important;
        border-right: 1px solid var(--color-splitter-border, #d7e1e2) !important;
      }

      /* vertical gutter */
      .gutter.gutter-vertical {
        background-color: var(--color-splitter-bg, #f0f0f0) !important;
        height: 10px !important;
        cursor: row-resize !important;
        border-top: 1px solid var(--color-splitter-border, #d7e1e2) !important;
        border-bottom: 1px solid var(--color-splitter-border, #d7e1e2) !important;
      }
    `;
    el.appendChild(style);

    // Now request fullscreen on that element
    if (el.requestFullscreen) {
      el.requestFullscreen().catch(() => {});
    }

    // Exit fullscreen on Escape
    const onKey = (e) => {
      if (e.key === 'Escape' && document.fullscreenElement) {
        document.exitFullscreen();
      }
    };
    document.addEventListener('keydown', onKey);

    return () => {
      document.removeEventListener('keydown', onKey);
      if (document.fullscreenElement) {
        document.exitFullscreen();
      }
      el.removeChild(style);
    };
  }, []);

  // Pre‑written C boilerplate to appear by default
  const defaultCode = `#include <assert.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* readline();
char* ltrim(char*);
char* rtrim(char*);

int parse_int(char*);

/*
 * Complete the 'fizzBuzz' function below.
 *
 * The function accepts INTEGER n as parameter.
 */

void fizzBuzz(int n) {

}

int main() {
    // Your code here
}
`;

  // Define custom light theme matching reference
  const handleEditorWillMount = (monaco) => {
    monaco.editor.defineTheme('customLight', {
      base: 'vs',
      inherit: true,
      rules: [
        { token: 'comment',    foreground: '008000' },
        { token: 'keyword',    foreground: '0000ff' },
        { token: 'string',     foreground: 'a31515' },
        { token: 'number',     foreground: '098658' },
        { token: 'identifier', foreground: '000000' },
      ],
      colors: {
        'editor.background':                '#ffffff',
        'editorLineNumber.foreground':      '#888888',
        'editorGutter.background':          '#ffffff',
        'editorLineNumber.activeForeground':'#000000',
      }
    });
  };

  return (
    <div ref={rootRef} style={{ display: 'flex', height: '100vh', width: '100vw' }}>
      {/* Sidebar */}
      <div
        style={{
          width: '80px',
          backgroundColor: '#e7eeef',
          boxShadow: 'inset 0 1px 4px rgba(57,66,78,0.1)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          flexShrink: 0,
        }}
      >
        {/* Timer & Icons */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {[
            { key: 'timer', content: '51m', bg: '#39424e', color: '#f3f7f7' },
            { key: 'eye',   content: '👁',  bg: '#e7eeef', color: '#0e141e' },
            { key: 'info',  content: 'ℹ️',  bg: '#e7eeef', color: '#0e141e' },
          ].map(item => (
            <button
              key={item.key}
              style={{
                width: '100%',
                height: '60px',
                backgroundColor: item.bg,
                color: item.color,
                fontSize: '20px',
                border: 'none',
                cursor: 'pointer',
              }}
            >
              {item.content}
            </button>
          ))}
        </div>
        {/* Question List */}
        <ul
          style={{
            margin: 0,
            padding: 0,
            listStyle: 'none',
            textAlign: 'center',
            flex: 1,
            overflowY: 'auto',
          }}
        >
          {QUESTIONS.map(q => (
            <li key={q.id}>
              <button
                onClick={() => setSelected(q)}
                style={{
                  width: '100%',
                  height: '60px',
                  border: 'none',
                  background:
                    q.id === selected.id ? '#fff' : 'transparent',
                  borderLeft:
                    q.id === selected.id
                      ? '3px solid #007acc'
                      : '3px solid transparent',
                  cursor: 'pointer',
                }}
              >
                {q.id}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Horizontal Split */}
      <Split
        sizes={[40, 60]}
        minSize={[200, 200]}
        gutterSize={10}
        gutterClassName="pane-splitter--horizontal"
        direction="horizontal"
        style={{ display: 'flex', flex: 1, height: '100%' }}
      >
        {/* Question Panel */}
        <div
          className="ps-content-wrapper-v0"
          style={{
            flex: 1,
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            overflowY: 'auto',
            backgroundColor: '#fff',
            fontFamily: 'Arial, Helvetica, sans-serif',
            color: '#39424e',
            lineHeight: '1.4em',
          }}
        >
          <h3
            style={{
              margin: 0,
              padding: '16px 12px 8px',
              fontSize: '20px',
              borderBottom: '2px solid #e7eeef',
            }}
          >
            {selected.title}
          </h3>
          <div style={{ padding: '8px 12px' }}>
            <p style={{ whiteSpace: 'pre-wrap', margin: '0 0 12px' }}>
              Given a number <em>n</em>, for each integer <em>i</em> in the range from <em>1</em> to <em>n</em> inclusive,
              print one value per line as follows:
            </p>
            <ul style={{ listStyleType: 'disc', paddingLeft: '20px', margin: '0 0 16px' }}>
              <li>If <em>i</em> is a multiple of both 3 and 5, print <em>FizzBuzz</em>.</li>
              <li>If <em>i</em> is a multiple of 3 (but not 5), print <em>Fizz</em>.</li>
              <li>If <em>i</em> is a multiple of 5 (but not 3), print <em>Buzz</em>.</li>
              <li>Otherwise, print the value of <em>i</em>.</li>
            </ul>
            <p style={{ fontWeight: 'bold', margin: '0 0 8px' }}>Function Description</p>
            <p style={{ margin: '0 0 12px' }}>
              Complete the function <em>fizzBuzz</em> in the editor below.
            </p>
            <p style={{ margin: '0 0 8px' }}><em>fizzBuzz</em> has the following parameter(s):</p>
            <ul style={{ listStyleType: 'none', paddingLeft: 0, margin: '0 0 16px' }}>
              <li style={{ margin: '4px 0' }}><em>int n:</em> upper limit of values to test</li>
            </ul>
            <p style={{ fontWeight: 'bold', margin: '0 0 8px' }}>Constraints</p>
            <ul style={{ listStyleType: 'disc', paddingLeft: '20px', margin: '0 0 16px' }}>
              <li><em>0 &lt; n &lt; 2 × 10<sup>5</sup></em></li>
            </ul>
            <details open style={{ margin: '0 0 16px' }}>
              <summary style={{
                backgroundColor: '#39424e',
                color: '#fff',
                fontWeight: 'bold',
                padding: '8px 12px',
                cursor: 'pointer',
                margin: 0,
              }}>
                Sample Case
              </summary>
              <pre style={{
                backgroundColor: '#f4faff',
                borderRadius: '2px',
                margin: '8px 12px',
                padding: '10px',
              }}>
1
2
Fizz
4
Buzz
Fizz
7
8
Fizz
Buzz
11
Fizz
13
14
FizzBuzz
              </pre>
            </details>
          </div>
        </div>

        {/* Bottom Split */}
        <Split
          sizes={[75, 25]}
          minSize={[200, 100]}
          gutterSize={10}
          gutterClassName="pane-splitter--horizontal"
          direction="vertical"
          style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
        >
          {/* Toolbar & Editor */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                padding: '0.5rem 1rem',
                background: '#fff',
                borderBottom: '1px solid #ddd',
              }}
            >
              <label style={{ marginRight: '0.5rem' }}>Language</label>
              <select
                style={{
                  padding: '0.25rem 0.5rem',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  marginRight: '1rem',
                }}
              >
                <option>C</option>
                <option>JavaScript (Node.js)</option>
                <option>Python 3</option>
                <option>C++14</option>
              </select>
              <span style={{ marginRight: 'auto' }}>Environment</span>
              <span style={{ color: '#28a745', marginRight: '1rem', fontSize: '0.9rem' }}>
                Autocomplete Ready
              </span>
              {['👁', '🌙', '⟳', '?'].map(icon => (
                <button
                  key={icon}
                  style={{
                    padding: '0.5rem',
                    marginLeft: '0.25rem',
                    border: 'none',
                    background: 'transparent',
                    cursor: 'pointer',
                  }}
                >
                  {icon}
                </button>
              ))}
            </div>
            <div style={{ flex: 1, background: '#fff' }}>
              <Editor
                height="100%"
                defaultLanguage="c"
                defaultValue={defaultCode}
                theme="customLight"
                beforeMount={handleEditorWillMount}
                options={{
                  minimap: { enabled: false },
                  scrollbar: { verticalScrollbarSize: 6, horizontalScrollbarSize: 6 },
                }}
              />
            </div>
          </div>

          {/* Bottom Tabs & Actions */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              background: '#f1f3f5',
              borderTop: '1px solid #ddd',
            }}
          >
            <div style={{ display: 'flex', marginLeft: '1rem' }}>
              {['results', 'custom'].map(tab => (
                <div
                  key={tab}
                  onClick={() => setBottomTab(tab)}
                  style={{
                    padding: '0.75rem 1rem',
                    cursor: 'pointer',
                    borderBottom:
                      bottomTab === tab ? '3px solid #007acc' : '3px solid transparent',
                    color: bottomTab === tab ? '#000' : '#666',
                  }}
                >
                  {tab === 'results' ? 'Test Results' : 'Custom Input'}
                </div>
              ))}
            </div>
            <div style={{ flex: 1 }} />
            <div style={{ marginRight: '1rem' }}>
              {['Run Code', 'Run Tests', 'Submit'].map((label, i) => (
                <button
                  key={i}
                  style={{
                    padding: '0.5rem 1rem',
                    marginLeft: '0.25rem',
                    border: 'none',
                    borderRadius: '4px',
                    background:
                      label === 'Submit'
                        ? '#28a745'
                        : label === 'Run Tests'
                        ? '#000'
                        : '#007acc',
                    color: '#fff',
                    cursor: 'pointer',
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </Split>
      </Split>
    </div>
  );
}
