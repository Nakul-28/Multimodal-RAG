"use client";

import { useCallback, useEffect, useState } from "react";
import { uploadFiles, connectStatusWS, fetchDocuments } from "@/lib/api";
import type { StatusUpdate, UploadJob, PipelineStage, DocumentInfo } from "@/lib/types";
import PipelineStatus from "./PipelineStatus";

const SUPPORTED_EXTENSIONS = [
  ".pdf", ".docx", ".doc", ".pptx", ".ppt",
  ".xlsx", ".xls", ".csv", ".tsv",
  ".txt", ".md", ".rst", ".rtf",
  ".html", ".htm", ".xml",
  ".eml", ".msg", ".epub",
  ".odt", ".org", ".json",
];

function isSupportedFile(name: string): boolean {
  return SUPPORTED_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext));
}

function getFileExt(name: string): string {
  const m = name.match(/\.([^.]+)$/);
  return m ? m[1].toUpperCase() : "FILE";
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function UploadPanel() {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [jobs, setJobs] = useState<UploadJob[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [docsError, setDocsError] = useState<string | null>(null);

  const loadDocuments = useCallback(async () => {
    setDocsLoading(true);
    setDocsError(null);
    try {
      const data = await fetchDocuments();
      setDocuments(data.documents ?? []);
    } catch (err) {
      setDocsError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setDocsLoading(false);
    }
  }, []);

  useEffect(() => { loadDocuments(); }, [loadDocuments]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);
  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => isSupportedFile(f.name));
    if (files.length) setSelectedFiles((prev) => [...prev, ...files]);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files).filter((f) => isSupportedFile(f.name));
      setSelectedFiles((prev) => [...prev, ...files]);
    }
    e.target.value = "";
  }, []);

  const removeFile = (idx: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleUpload = async () => {
    if (!selectedFiles.length) return;
    setError(null);
    setUploading(true);
    try {
      const resp = await uploadFiles(selectedFiles);
      const jobId = resp.job_id;
      const filenames = resp.filenames || [resp.filename || "unknown"];
      const newJob: UploadJob = { jobId, filenames, status: null };
      setJobs((prev) => [newJob, ...prev]);
      setSelectedFiles([]);
      connectStatusWS(
        jobId,
        (data) => {
          const update = data as unknown as StatusUpdate;
          setJobs((prev) =>
            prev.map((j) => (j.jobId === jobId ? { ...j, status: update } : j))
          );
        },
        () => { loadDocuments(); }
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div
      className="animate-fade-up delay-100"
      style={{ display: "flex", flexDirection: "column" }}
    >
      {/* ══ INGEST SECTION ══ */}
      <div style={{ flexShrink: 0, borderBottom: "2px solid var(--border)" }}>
        {/* Header */}
        <div style={{ padding: "0.9rem 1.5rem 0.6rem", borderBottom: "1px solid var(--border-light)" }}>
          <div className="ornament">Ingest</div>
        </div>

        {/* Content */}
        <div style={{ padding: "1rem 1.25rem", display: "flex", flexDirection: "column", gap: "0.9rem" }}>
          {/* Drop Zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById("file-input")?.click()}
            className={`drop-zone${dragOver ? " drag-over" : ""}`}
            style={{ padding: "1.4rem 1rem", textAlign: "center", cursor: "pointer" }}
          >
            <div
              style={{
                width: "36px",
                height: "36px",
                border: "1px solid var(--border-light)",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 0.75rem",
              }}
            >
              <svg width="14" height="14" fill="none" stroke="var(--ink-3)" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            </div>
            <p style={{ fontFamily: "var(--font-body)", fontSize: "0.875rem", color: "var(--ink-2)", margin: 0 }}>
              Drop files or{" "}
              <span style={{ color: "var(--gold)", fontStyle: "italic" }}>browse</span>
            </p>
            <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--ink-3)", letterSpacing: "0.06em", marginTop: "0.4rem" }}>
              PDF &middot; DOCX &middot; TXT &middot; PPTX &middot; XLSX &middot; CSV &middot; HTML &middot; MD
            </p>
            <input
              id="file-input"
              type="file"
              accept={SUPPORTED_EXTENSIONS.join(",")}
              multiple
              style={{ display: "none" }}
              onChange={handleFileInput}
            />
          </div>

          {/* Queued files */}
          {selectedFiles.length > 0 && (
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "var(--ink-3)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "0.5rem" }}>
                Queued &mdash; {selectedFiles.length}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
                {selectedFiles.map((f, i) => (
                  <div
                    key={i}
                    style={{ display: "flex", alignItems: "center", gap: "0.5rem", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "2px", padding: "0.4em 0.6em" }}
                  >
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.55rem", color: "var(--gold)", letterSpacing: "0.05em", minWidth: "30px", textAlign: "center", border: "1px solid var(--gold)", borderRadius: "1px", padding: "0 2px" }}>
                      {getFileExt(f.name)}
                    </span>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-2)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {f.name}
                    </span>
                    <button
                      onClick={() => removeFile(i)}
                      style={{ background: "none", border: "none", color: "var(--ink-3)", cursor: "pointer", fontSize: "0.75rem", lineHeight: 1, padding: "2px", transition: "color 0.15s" }}
                      onMouseOver={(e) => (e.currentTarget.style.color = "var(--error)")}
                      onMouseOut={(e) => (e.currentTarget.style.color = "var(--ink-3)")}
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="btn-gold"
                style={{ marginTop: "0.75rem", width: "100%" }}
              >
                {uploading ? "Ingesting\u2026" : `Ingest ${selectedFiles.length} file${selectedFiles.length > 1 ? "s" : ""}`}
              </button>
            </div>
          )}

          {/* Upload error */}
          {error && (
            <div style={{ background: "var(--error-dim)", border: "1px solid var(--error)", borderRadius: "2px", padding: "0.5em 0.75em", fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--error)" }}>
              {error}
            </div>
          )}

          {/* Processing jobs */}
          {jobs.length > 0 && (
            <div>
              <div className="ornament" style={{ marginBottom: "0.75rem" }}>Processing</div>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                {jobs.map((job) => (
                  <div key={job.jobId} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "3px", padding: "0.75rem" }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ink-2)", marginBottom: "0.6rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {job.filenames.join(", ")}
                    </div>
                    {job.status ? (
                      <PipelineStatus
                        stage={job.status.stage as PipelineStage}
                        message={job.status.message}
                        filesProcessed={job.status.files_processed}
                        totalFiles={job.status.total_files}
                        currentFile={job.status.current_file}
                      />
                    ) : (
                      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ink-3)", display: "flex", alignItems: "center", gap: "0.4rem" }}>
                        <span style={{ animation: "spinSlow 1.5s linear infinite", display: "inline-block" }}>&#9676;</span>
                        Queued&hellip;
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ══ CORPUS SECTION ══ */}
      <div style={{ display: "flex", flexDirection: "column" }}>
        {/* Header */}
        <div
          style={{
            padding: "0.9rem 1.5rem 0.6rem",
            borderBottom: "1px solid var(--border-light)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexShrink: 0,
          }}
        >
          <div className="ornament" style={{ flex: 1 }}>
            Corpus{documents.length > 0 ? ` \u00b7 ${documents.length}` : ""}
          </div>
          <button
            onClick={loadDocuments}
            title="Refresh"
            style={{ background: "none", border: "none", color: "var(--ink-3)", cursor: "pointer", fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.08em", padding: "0.1em 0.3em", marginLeft: "0.5rem", transition: "color 0.15s" }}
            onMouseOver={(e) => (e.currentTarget.style.color = "var(--gold)")}
            onMouseOut={(e) => (e.currentTarget.style.color = "var(--ink-3)")}
          >
            {docsLoading ? "..." : "\u21bb"}
          </button>
        </div>

        {/* Document list */}
        <div style={{ padding: "1rem 1.25rem", display: "flex", flexDirection: "column", gap: "0.3rem" }}>
          {docsLoading && (
            <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--ink-3)", fontStyle: "italic" }}>
              Loading&hellip;
            </p>
          )}

          {!docsLoading && docsError && (
            <div style={{ background: "var(--error-dim)", border: "1px solid var(--error)", borderRadius: "2px", padding: "0.5em 0.75em", fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--error)" }}>
              {docsError}
            </div>
          )}

          {!docsLoading && !docsError && documents.length === 0 && (
            <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--ink-3)", fontStyle: "italic" }}>
              No documents ingested yet.
            </p>
          )}

          {!docsLoading && documents.length > 0 && documents.map((doc) => (
            <div
              key={doc.full_path}
              style={{ display: "flex", alignItems: "flex-start", gap: "0.6rem", padding: "0.5em 0.6em", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "2px" }}
            >
              <div
                style={{ width: "14px", height: "14px", border: "1px solid var(--success)", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: "1px" }}
              >
                <svg width="6" height="6" viewBox="0 0 6 6" fill="none" stroke="var(--success)" strokeWidth="1.5">
                  <path d="M1 3 L2.5 4.5 L5 1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", margin: 0 }}>
                  {doc.filename}
                </p>
                <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.58rem", color: "var(--ink-3)", margin: "0.15em 0 0" }}>
                  {formatFileSize(doc.size_bytes)} &middot; {formatDate(doc.uploaded_at)}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

