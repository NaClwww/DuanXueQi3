export const recognize: (
  param: ArrayBuffer,
  model: ArrayBuffer,
  rgba?: ArrayBuffer,
  width?: number,
  height?: number,
  labels?: ArrayBuffer,
  detParam?: ArrayBuffer,
  detModel?: ArrayBuffer,
  recParam?: ArrayBuffer,
  recModel?: ArrayBuffer,
  useGpu?: boolean
) => string;

export const getRuntimeInfo: (useGpu?: boolean) => string;
