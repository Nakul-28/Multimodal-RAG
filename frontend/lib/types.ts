// TypeScript types for RAG Pipeline frontend

export type PipelineStage =
  | "pending"
  | "parsing"
  | "chunking"
  | "summarizing"
  | "embedding"
  | "done"
  | "failed";

export interface StatusUpdate {
  status: string;
  stage: PipelineStage;
  message: string;
  files_processed: number;
  total_files: number;
  current_file: string;
  error?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  images?: string[];
  isStreaming?: boolean;
}

export interface UploadResponse {
  job_id: string;
  filename?: string;
  total_files?: number;
  filenames?: string[];
  status: string;
}

export interface HealthResponse {
  status: string;
  vector_store_documents: number;
  images_in_memory: number;
  active_jobs: number;
}

export interface SSEMetadata {
  type: "metadata";
  retrieved_chunks: number;
  image_ids: string[];
}

export interface SSEToken {
  type: "token";
  content: string;
}

export interface SSEError {
  type: "error";
  content: string;
}

export type SSEMessage = SSEMetadata | SSEToken | SSEError;

export interface UploadJob {
  jobId: string;
  filenames: string[];
  status: StatusUpdate | null;
}

export interface DocumentInfo {
  filename: string;
  size_bytes: number;
  uploaded_at: number;
  full_path: string;
}
