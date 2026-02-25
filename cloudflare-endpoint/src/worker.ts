import {
  type GenerateSampleInput,
  generateSample,
} from "../../lib/generate-sample.ts"

type KVNamespace = {
  get: (key: string, type: "json") => Promise<unknown>
  put: (key: string, value: string) => Promise<void>
}

type Env = {
  SAMPLE_CACHE: KVNamespace
}

type GenerateRequest = {
  problemId: string
  seed: number
  pairCount?: number
  minPointSeparationMm?: number
  cellSizeMm?: number
  viaDiameterMm?: number
  traceThicknessMm?: number
  traceMarginMm?: number
  maxSolveAttempts?: number
}

type CachedResponse = {
  ok: boolean
  cached: boolean
  sample?: {
    boundary_connection_pairs: unknown[]
    routed_paths: unknown[]
    connection_pair_svg: string
    routed_svg: string
  }
  reason?: string
  attempts: number
  usedSeed?: number
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url)

    if (request.method === "GET" && url.pathname === "/health") {
      return json({ ok: true, service: "sample-generator" })
    }

    if (request.method !== "POST" || url.pathname !== "/generate") {
      return json({ ok: false, reason: "Not found" }, 404)
    }

    let payload: GenerateRequest
    try {
      payload = (await request.json()) as GenerateRequest
    } catch {
      return json({ ok: false, reason: "Invalid JSON body" }, 400)
    }

    if (!payload.problemId || !Number.isFinite(payload.seed)) {
      return json(
        {
          ok: false,
          reason: "`problemId` (string) and `seed` (number) are required",
        },
        400,
      )
    }

    const normalizedInput: GenerateSampleInput = {
      problemId: payload.problemId,
      seed: Math.floor(payload.seed),
      pairCount:
        payload.pairCount !== undefined
          ? Math.floor(payload.pairCount)
          : undefined,
      minPointSeparationMm: payload.minPointSeparationMm,
      cellSizeMm: payload.cellSizeMm,
      viaDiameterMm: payload.viaDiameterMm,
      traceThicknessMm: payload.traceThicknessMm,
      traceMarginMm: payload.traceMarginMm,
      maxSolveAttempts: payload.maxSolveAttempts,
    }

    const cacheHash = await sha256Hex(stableStringify(normalizedInput))
    const cacheKey = `v4:${cacheHash}`
    const cached = await env.SAMPLE_CACHE.get(cacheKey, "json")

    if (cached) {
      const cachedResponse = cached as CachedResponse
      return json({ ...cachedResponse, cached: true })
    }

    const result = await generateSample(normalizedInput)
    const responseBody: CachedResponse = result.ok
      ? {
          ok: true,
          cached: false,
          sample: {
            boundary_connection_pairs: result.sample.boundary_connection_pairs,
            routed_paths: result.sample.routed_paths,
            connection_pair_svg: result.sample.connection_pair_svg,
            routed_svg: result.sample.routed_svg,
          },
          attempts: result.attempts,
          usedSeed: result.usedSeed,
        }
      : {
          ok: false,
          cached: false,
          reason: result.reason,
          attempts: result.attempts,
        }

    await env.SAMPLE_CACHE.put(cacheKey, JSON.stringify(responseBody))

    return json(responseBody)
  },
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
    },
  })
}

function stableStringify(value: unknown): string {
  return JSON.stringify(sortValue(value))
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortValue)
  }

  if (value && typeof value === "object") {
    const sortedEntries = Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, childValue]) => [key, sortValue(childValue)])
    return Object.fromEntries(sortedEntries)
  }

  return value
}

async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input)
  const digest = await crypto.subtle.digest("SHA-256", bytes)
  const hashBytes = new Uint8Array(digest)
  return Array.from(hashBytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("")
}
