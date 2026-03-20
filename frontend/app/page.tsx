"use client";

import { useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import ChatWindow from "@/components/ChatWindow";

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
        background: "var(--bg)",
      }}
    >
      {/* ─── Mobile toggle ─── */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{
          position: "fixed",
          top: "1rem",
          left: "1rem",
          zIndex: 50,
          padding: "0.45rem",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          color: "var(--ink-2)",
          borderRadius: "3px",
          cursor: "pointer",
          display: "none",
        }}
        className="mobile-toggle"
        aria-label="Toggle sidebar"
      >
        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {sidebarOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>

      {/* ─── Collapse / Expand toggle button (sits on the sidebar border) ─── */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        style={{
          position: "fixed",
          left: collapsed ? "0px" : "300px",
          top: "50%",
          transform: "translateY(-50%)",
          zIndex: 40,
          width: "18px",
          height: "48px",
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderLeft: collapsed ? "1px solid var(--border)" : "none",
          borderRadius: collapsed ? "0 4px 4px 0" : "0 4px 4px 0",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--ink-3)",
          transition: "left 0.28s ease",
          padding: 0,
        }}
        onMouseOver={(e) => (e.currentTarget.style.color = "var(--gold)")}
        onMouseOut={(e) => (e.currentTarget.style.color = "var(--ink-3)")}
      >
        <svg
          width="8"
          height="14"
          viewBox="0 0 8 14"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transition: "transform 0.3s ease", transform: collapsed ? "rotate(180deg)" : "rotate(0deg)" }}
        >
          <path d="M6 1L2 7l4 6" />
        </svg>
      </button>

      {/* ─── Sidebar / Archive Panel ─── */}
      <aside
        className="animate-fade-up"
        style={{
          width: collapsed ? "0px" : "300px",
          minWidth: collapsed ? "0px" : "300px",
          background: "var(--panel)",
          borderRight: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          position: "relative",
          overflow: "hidden",
          animationDelay: "0s",
          transition: "width 0.28s ease, min-width 0.28s ease",
        }}
      >
        {/* Ambient glow top */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: "50%",
            transform: "translateX(-50%)",
            width: "160px",
            height: "80px",
            background: "radial-gradient(ellipse at top, rgba(201,165,90,0.08) 0%, transparent 70%)",
            pointerEvents: "none",
          }}
        />

        {/* App Identity */}
        <div
          style={{
            padding: "1.75rem 1.5rem 1.25rem",
            borderBottom: "1px solid var(--border)",
            position: "relative",
            flexShrink: 0,
          }}
        >
          {/* Small ornament */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              marginBottom: "0.75rem",
            }}
          >
            <div
              style={{
                width: "20px",
                height: "20px",
                border: "1px solid var(--gold)",
                borderRadius: "2px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="var(--gold)">
                <path d="M5 1 L6.2 3.8 L9 4.2 L7 6.2 L7.5 9 L5 7.6 L2.5 9 L3 6.2 L1 4.2 L3.8 3.8 Z"/>
              </svg>
            </div>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.6rem",
                letterSpacing: "0.2em",
                textTransform: "uppercase",
                color: "var(--gold)",
              }}
            >
              Intel System
            </span>
          </div>

          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.45rem",
              fontWeight: 600,
              color: "var(--ink)",
              lineHeight: 1.15,
              letterSpacing: "-0.01em",
              margin: 0,
            }}
          >
            RAG Pipeline
          </h1>
          <p
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.65rem",
              color: "var(--ink-3)",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              marginTop: "0.3rem",
            }}
          >
            Multimodal Document Q&amp;A
          </p>
        </div>

        {/* Upload panel — scrollable */}
        <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>
          <UploadPanel />
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.6)",
            zIndex: 30,
          }}
          className="md-hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ─── Main Chat Area ─── */}
      <main
        className="animate-fade-in delay-200"
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          background: "var(--bg)",
          position: "relative",
        }}
      >
        {/* Ambient glow bottom-right */}
        <div
          style={{
            position: "absolute",
            bottom: "10%",
            right: "5%",
            width: "300px",
            height: "300px",
            background: "radial-gradient(ellipse, rgba(201,165,90,0.03) 0%, transparent 70%)",
            pointerEvents: "none",
          }}
        />
        <ChatWindow />
      </main>
    </div>
  );
}

