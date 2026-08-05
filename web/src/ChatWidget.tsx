import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import { ChatApiError, sendChatMessage } from './api';
import { getOrCreateThreadId, resetThreadId } from './threadId';
import type { ChatMessage } from './types';
import './ChatWidget.css';

type ConnectionStatus = 'checking' | 'online' | 'offline';

function makeId(): string {
  return crypto.randomUUID();
}

function formatTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function TypingIndicator() {
  return (
    <div className="bubble-row assistant">
      <div className="bubble assistant typing" aria-label="Assistant is typing">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  );
}

function StatusDot({ status }: { status: ConnectionStatus }) {
  return <span className={`status-dot ${status}`} title={status} />;
}

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [threadId, setThreadId] = useState<string>(() => getOrCreateThreadId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [status, setStatus] = useState<ConnectionStatus>('checking');

  const listRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!open) return;

    let cancelled = false;
    setStatus('checking');

    fetch('/health')
      .then((res) => {
        if (!cancelled) setStatus(res.ok ? 'online' : 'offline');
      })
      .catch(() => {
        if (!cancelled) setStatus('offline');
      });

    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    textareaRef.current?.focus();
  }, [open]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, isSending]);

  useEffect(() => {
    if (!open) return undefined;

    const onKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open]);

  const handleNewConversation = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setThreadId(resetThreadId());
    setIsSending(false);
  }, []);

  const submitMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isSending) return;

      const userMessage: ChatMessage = {
        id: makeId(),
        role: 'user',
        text: trimmed,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setInput('');
      setIsSending(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const data = await sendChatMessage(
          { message: trimmed, thread_id: threadId },
          controller.signal,
        );
        setStatus('online');
        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: 'assistant',
            text: data.response,
            timestamp: data.timestamp,
            modelUsed: data.model_used,
            cached: data.cached,
            processingTimeMs: data.processing_time_ms,
          },
        ]);
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;

        const message =
          err instanceof ChatApiError ? err.message : 'Something went wrong. Please try again.';
        if (err instanceof ChatApiError && err.status === 0) setStatus('offline');

        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: 'assistant',
            text: message,
            timestamp: new Date().toISOString(),
            isError: true,
          },
        ]);
      } finally {
        setIsSending(false);
        abortRef.current = null;
      }
    },
    [isSending, threadId],
  );

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submitMessage(input);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void submitMessage(input);
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  };

  return (
    <>
      <button
        type="button"
        className={`launcher ${open ? 'hidden' : ''}`}
        onClick={() => setOpen(true)}
        aria-label="Open chat"
      >
        <svg viewBox="0 0 24 24" width="26" height="26" fill="none" aria-hidden="true">
          <path
            d="M4 4h16v12H8l-4 4V4z"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </svg>
      </button>

      {open && (
        <div className="backdrop" onClick={() => setOpen(false)}>
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="ChatAgent tester"
            onClick={(e) => e.stopPropagation()}
          >
            <header className="modal-header">
              <div className="modal-header-title">
                <span className="bot-avatar">✦</span>
                <div>
                  <h2>ChatAgent</h2>
                  <div className="subtitle">
                    <StatusDot status={status} />
                    {status === 'checking' && 'Connecting…'}
                    {status === 'online' && 'Online'}
                    {status === 'offline' && 'Backend unreachable'}
                  </div>
                </div>
              </div>
              <div className="modal-header-actions">
                <button
                  type="button"
                  className="icon-button"
                  title="New conversation"
                  aria-label="New conversation"
                  onClick={handleNewConversation}
                >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
                    <path
                      d="M4 12a8 8 0 1 1 2.34 5.66M4 12V6m0 6h6"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>
                <button
                  type="button"
                  className="icon-button"
                  title="Close"
                  aria-label="Close chat"
                  onClick={() => setOpen(false)}
                >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
                    <path
                      d="M6 6l12 12M18 6L6 18"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                    />
                  </svg>
                </button>
              </div>
            </header>

            <div className="message-list" ref={listRef}>
              {messages.length === 0 && (
                <div className="empty-state">
                  <span className="bot-avatar large">✦</span>
                  <p>Ask me anything — I&apos;m wired up to the live API.</p>
                </div>
              )}

              {messages.map((m) => (
                <div key={m.id} className={`bubble-row ${m.role}`}>
                  <div className={`bubble ${m.role} ${m.isError ? 'error' : ''}`}>
                    <div className="bubble-text">{m.text}</div>
                    <div className="bubble-meta">
                      {m.role === 'assistant' && !m.isError && (
                        <>
                          {m.modelUsed && <span className="chip">{m.modelUsed}</span>}
                          {m.cached && <span className="chip cached">cached</span>}
                          {typeof m.processingTimeMs === 'number' && !m.cached && (
                            <span className="chip">{Math.round(m.processingTimeMs)}ms</span>
                          )}
                        </>
                      )}
                      <span className="time">{formatTime(m.timestamp)}</span>
                    </div>
                  </div>
                </div>
              ))}

              {isSending && <TypingIndicator />}
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
                rows={1}
                disabled={isSending}
              />
              <button
                type="submit"
                className="send-button"
                disabled={isSending || input.trim().length === 0}
                aria-label="Send message"
              >
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
                  <path
                    d="M4 12l16-7-6 16-2.5-6.5L4 12z"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
