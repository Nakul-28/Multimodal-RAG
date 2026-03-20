"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { clearChatHistory, streamChat, imageUrl } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";

// ─── Empty State ────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        padding: "3rem",
        textAlign: "center",
        userSelect: "none",
      }}
    >
      {/* Decorative ring */}
      <div
        className="animate-fade-up"
        style={{
          position: "relative",
          width: "80px",
          height: "80px",
          marginBottom: "2rem",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            border: "1px solid var(--border-light)",
            borderRadius: "50%",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: "10px",
            border: "1px dashed var(--border)",
            borderRadius: "50%",
          }}
        />
        <div
          style={{
            position: "absolute",
            inset: "50%",
            transform: "translate(-50%, -50%)",
            width: "10px",
            height: "10px",
            background: "var(--gold)",
            borderRadius: "50%",
            opacity: 0.8,
          }}
        />
        {/* Ornamental ticks */}
        {[0, 90, 180, 270].map((deg) => (
          <div
            key={deg}
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              width: "6px",
              height: "1px",
              background: "var(--gold)",
              opacity: 0.5,
              transformOrigin: "left center",
              transform: `rotate(${deg}deg) translateX(30px) translateY(-50%)`,
            }}
          />
        ))}
      </div>

      <h2
        className="animate-fade-up delay-100"
        style={{
          fontFamily: "var(--font-display)",
          fontSize: "1.6rem",
          fontWeight: 500,
          color: "var(--ink)",
          letterSpacing: "-0.01em",
          lineHeight: 1.2,
          margin: 0,
        }}
      >
        Interrogate your archive
      </h2>
      <p
        className="animate-fade-up delay-200"
        style={{
          fontFamily: "var(--font-body)",
          fontSize: "0.95rem",
          color: "var(--ink-3)",
          fontStyle: "italic",
          marginTop: "0.6rem",
          maxWidth: "280px",
          lineHeight: 1.5,
        }}
      >
        Upload documents in the panel, then ask anything you wish to know.
      </p>

      {/* Suggestion chips */}
      <div
        className="animate-fade-up delay-300"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.5rem",
          marginTop: "2rem",
          justifyContent: "center",
          maxWidth: "400px",
        }}
      >
        {[
          "Summarise the key findings",
          "What are the main risks?",
          "Compare the documents",
          "Extract all dates and figures",
        ].map((suggestion) => (
          <span
            key={suggestion}
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.68rem",
              color: "var(--ink-3)",
              border: "1px solid var(--border)",
              borderRadius: "2px",
              padding: "0.3em 0.75em",
              letterSpacing: "0.04em",
            }}
          >
            {suggestion}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Message component ───────────────────────────────────────────────────────
function MessageItem({ msg, index }: { msg: ChatMessage; index: number }) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div
        className="animate-fade-up"
        style={{
          animationDelay: `${Math.min(index * 0.04, 0.3)}s`,
          display: "flex",
          justifyContent: "flex-end",
          padding: "0.5rem 0",
        }}
      >
        <div style={{ maxWidth: "62%", textAlign: "right" }}>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.62rem",
              color: "var(--ink-3)",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              marginBottom: "0.3rem",
            }}
          >
            You
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.8125rem",
              color: "var(--ink-2)",
              lineHeight: 1.55,
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "3px 3px 0 3px",
              padding: "0.65em 0.9em",
              display: "inline-block",
              textAlign: "left",
              maxWidth: "100%",
              wordBreak: "break-word",
            }}
          >
            <span
              style={{
                color: "var(--gold)",
                marginRight: "0.4em",
                fontWeight: 500,
              }}
            >
              ›
            </span>
            {msg.content}
          </div>
        </div>
      </div>
    );
  }

  // Assistant
  return (
    <div
      className="animate-fade-up"
      style={{
        animationDelay: `${Math.min(index * 0.04, 0.3)}s`,
        padding: "1rem 0",
        borderBottom: "1px solid var(--border)",
      }}
    >
      {/* Label row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.6rem",
          marginBottom: "0.75rem",
        }}
      >
        <div
          style={{
            width: "18px",
            height: "18px",
            border: "1px solid var(--gold)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <svg width="8" height="8" viewBox="0 0 8 8" fill="var(--gold)">
            <path d="M4 0.5 L4.9 3 L7.5 3.2 L5.6 4.9 L6.2 7.5 L4 6.2 L1.8 7.5 L2.4 4.9 L0.5 3.2 L3.1 3 Z"/>
          </svg>
        </div>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.62rem",
            color: "var(--gold)",
            letterSpacing: "0.14em",
            textTransform: "uppercase",
          }}
        >
          Intelligence
        </span>
        {msg.isStreaming && (
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.6rem",
              color: "var(--ink-3)",
              letterSpacing: "0.1em",
              fontStyle: "italic",
            }}
          >
            composing…
          </span>
        )}
      </div>

      {/* Body */}
      <div
        className="msg-prose"
        style={{
          paddingLeft: "1.65rem",
          fontFamily: "var(--font-body)",
          fontSize: "1rem",
          color: "var(--ink)",
          lineHeight: 1.72,
        }}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {msg.content}
        </ReactMarkdown>
        {msg.isStreaming && <span className="cursor-blink" />}
      </div>

      {/* Images */}
      {msg.images && msg.images.length > 0 && (
        <div
          style={{
            paddingLeft: "1.65rem",
            marginTop: "1rem",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
            gap: "0.5rem",
          }}
        >
          {msg.images.map((imgId) => (
            <div
              key={imgId}
              style={{
                border: "1px solid var(--border)",
                borderRadius: "3px",
                overflow: "hidden",
                cursor: "pointer",
              }}
              onClick={() => window.open(imageUrl(imgId), "_blank")}
            >
              <img
                src={imageUrl(imgId)}
                alt="Document image"
                style={{
                  width: "100%",
                  maxHeight: "160px",
                  objectFit: "contain",
                  display: "block",
                  background: "var(--surface)",
                  transition: "opacity 0.15s ease",
                }}
                onMouseOver={(e) => (e.currentTarget.style.opacity = "0.8")}
                onMouseOut={(e) => (e.currentTarget.style.opacity = "1")}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Chat Window ─────────────────────────────────────────────────────────────
export default function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 130)}px`;
    }
  };

  const handleSend = async () => {
    const query = input.trim();
    if (!query || isStreaming) return;

    const userMsg: ChatMessage = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";

    const assistantMsg: ChatMessage = {
      role: "assistant",
      content: "",
      images: [],
      isStreaming: true,
    };
    setMessages((prev) => [...prev, assistantMsg]);
    setIsStreaming(true);

    try {
      let accum = "";
      let images: string[] = [];

      for await (const event of streamChat(query)) {
        if (event.type === "metadata") {
          images = event.image_ids || [];
        } else if (event.type === "token") {
          accum += event.content;
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: "assistant",
              content: accum,
              images,
              isStreaming: true,
            };
            return updated;
          });
        } else if (event.type === "error") {
          accum += `\n\n**Error:** ${event.content}`;
        }
      }

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: accum || "No response received.",
          images,
          isStreaming: false,
        };
        return updated;
      });
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `**Error:** ${err instanceof Error ? err.message : "Something went wrong"}`,
          images: [],
          isStreaming: false,
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = async () => {
    try {
      await clearChatHistory();
    } catch {
      // Keep the UI responsive even if the backend clear request fails.
    } finally {
      setMessages([]);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        position: "relative",
      }}
    >
      {/* ── Header bar ── */}
      <div
        style={{
          padding: "0.9rem 2rem",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
        }}
      >
        <div className="ornament" style={{ flex: 1, maxWidth: "200px" }}>
          Chat
        </div>
        {messages.length > 0 && (
          <button
            onClick={handleClear}
            title="Clear chat history"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.65rem",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--ink-3)",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "3px",
              cursor: "pointer",
              padding: "0.35em 0.7em",
              transition: "all 0.15s",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.color = "var(--error)";
              e.currentTarget.style.borderColor = "var(--error)";
              e.currentTarget.style.background = "var(--error-dim)";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.color = "var(--ink-3)";
              e.currentTarget.style.borderColor = "var(--border)";
              e.currentTarget.style.background = "var(--surface)";
            }}
          >
            <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Clear History
          </button>
        )}
      </div>

      {/* ── Messages ── */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "1.5rem 2rem",
        }}
      >
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageItem key={i} msg={msg} index={i} />
            ))}
            <div ref={messagesEndRef} style={{ height: "1px" }} />
          </>
        )}
      </div>

      {/* ── Input ── */}
      <div
        style={{
          borderTop: "1px solid var(--border)",
          padding: "1.25rem 2rem 1.5rem",
          flexShrink: 0,
          position: "relative",
        }}
      >
        {/* Gold hint line */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: "2rem",
            right: "2rem",
            height: "1px",
            background: "linear-gradient(90deg, var(--gold) 0%, transparent 60%)",
            opacity: 0.3,
          }}
        />

        <div
          style={{
            maxWidth: "780px",
            margin: "0 auto",
            display: "flex",
            flexDirection: "column",
            gap: "0.9rem",
          }}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents…"
            rows={1}
            disabled={isStreaming}
            className="chat-input"
            style={{ display: "block" }}
          />

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.6rem",
                color: "var(--ink-3)",
                letterSpacing: "0.1em",
              }}
            >
              {isStreaming ? (
                <span style={{ color: "var(--gold)", animation: "shimmer 1.5s ease-in-out infinite" }}>
                  ◆ generating…
                </span>
              ) : (
                "⏎ send · shift+⏎ newline"
              )}
            </span>

            <button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="btn-gold"
            >
              {isStreaming ? "…" : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}