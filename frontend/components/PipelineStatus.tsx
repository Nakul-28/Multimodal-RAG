"use client";

import { PipelineStage } from "@/lib/types";

const STAGES: { key: PipelineStage; label: string; roman: string }[] = [
  { key: "parsing",    label: "Parse",   roman: "I"   },
  { key: "chunking",   label: "Chunk",   roman: "II"  },
  { key: "summarizing",label: "Summarise",roman: "III"},
  { key: "embedding",  label: "Embed",   roman: "IV"  },
  { key: "done",       label: "Done",    roman: "V"   },
];

function stageIndex(stage: PipelineStage): number {
  return STAGES.findIndex((s) => s.key === stage);
}

interface Props {
  stage: PipelineStage;
  message: string;
  filesProcessed: number;
  totalFiles: number;
  currentFile: string;
}

export default function PipelineStatus({
  stage,
  message,
  filesProcessed,
  totalFiles,
  currentFile,
}: Props) {
  const activeIdx = stageIndex(stage);
  const isFailed  = stage === "failed";
  const isDone    = stage === "done";

  return (
    <div style={{ width: "100%" }}>
      {/* File counter */}
      {totalFiles > 0 && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.6rem",
            color: "var(--ink-3)",
            letterSpacing: "0.08em",
            marginBottom: "0.6rem",
          }}
        >
          {Math.min(filesProcessed + (isDone ? 0 : 1), totalFiles)}/{totalFiles}
          {currentFile && (
            <span style={{ marginLeft: "0.5rem", color: "var(--ink-3)" }}>
              &mdash; {currentFile}
            </span>
          )}
        </div>
      )}

      {/* Stage rail */}
      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        {STAGES.map((s, idx) => {
          const isCompleted = !isFailed && (activeIdx > idx || (isDone && activeIdx === idx));
          const isActive    = !isFailed && !isDone && activeIdx === idx;
          const isThisFailed= isFailed && activeIdx === idx;
          const isPending   = !isFailed && activeIdx < idx;

          let dotColor = "var(--border-light)";
          let labelColor = "var(--ink-3)";
          if (isCompleted)   { dotColor = "var(--gold)";    labelColor = "var(--gold)"; }
          if (isActive)      { dotColor = "var(--gold)";    labelColor = "var(--ink)";  }
          if (isThisFailed)  { dotColor = "var(--error)";   labelColor = "var(--error)";}

          return (
            <div
              key={s.key}
              style={{
                display: "flex",
                alignItems: "flex-start",
                gap: "0.55rem",
                position: "relative",
              }}
            >
              {/* Vertical track + dot */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  flexShrink: 0,
                  marginTop: "2px",
                }}
              >
                {/* Dot */}
                <div
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    background: dotColor,
                    flexShrink: 0,
                    transition: "background 0.3s ease",
                    boxShadow: isActive
                      ? "0 0 6px rgba(201,165,90,0.5)"
                      : "none",
                    animation: isActive ? "shimmer 1.4s ease-in-out infinite" : "none",
                  }}
                />
                {/* Connector line */}
                {idx < STAGES.length - 1 && (
                  <div
                    style={{
                      width: "1px",
                      height: "14px",
                      background: isCompleted ? "var(--gold)" : "var(--border)",
                      opacity: isCompleted ? 0.5 : 1,
                      transition: "background 0.3s ease",
                      marginTop: "1px",
                    }}
                  />
                )}
              </div>

              {/* Label */}
              <div
                style={{
                  paddingBottom: idx < STAGES.length - 1 ? "0" : "0",
                  lineHeight: 1,
                  paddingTop: "0",
                  marginTop: "-1px",
                }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.65rem",
                    color: labelColor,
                    letterSpacing: "0.06em",
                    transition: "color 0.3s ease",
                    fontWeight: isActive ? 500 : 400,
                  }}
                >
                  {isCompleted ? "\u2713 " : isThisFailed ? "\u2717 " : ""}
                  {s.label}
                  {isActive && (
                    <span
                      style={{
                        marginLeft: "0.4rem",
                        color: "var(--ink-3)",
                        fontStyle: "italic",
                      }}
                    >
                      ...
                    </span>
                  )}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Status message */}
      {message && (
        <p
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.6rem",
            color: "var(--ink-3)",
            fontStyle: "italic",
            marginTop: "0.5rem",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {message}
        </p>
      )}
    </div>
  );
}
