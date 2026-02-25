import { mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { Resvg } from "@resvg/resvg-js"
import {
  CELL_SIZE_MM,
  IMAGE_SIZE_PX,
  VIA_DIAMETER_MM,
  TRACE_THICKNESS_MM,
  TRACE_MARGIN_MM,
  MAX_SOLVE_ATTEMPTS,
} from "../lib/generator-params.ts"

type GeneratedSampleResponse = {
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

type DatasetRow = {
  id: string
  boundary_connection_pairs: unknown[]
  connection_pair_image: string
  routed_image: string
  routed_paths: unknown[]
}

const { values: args } = parseArgs({
  options: {
    "sample-count": { type: "string" },
    endpoint: { type: "string" },
    "output-dir": { type: "string" },
    concurrency: { type: "string" },
  },
  strict: true,
})

const requiredFlags = {
  "--sample-count": args["sample-count"],
  "--output-dir": args["output-dir"],
  "--concurrency": args.concurrency,
} as const

const missing = Object.entries(requiredFlags)
  .filter(([, v]) => !v)
  .map(([k]) => k)

if (missing.length > 0) {
  console.error(
    `Missing required flags: ${missing.join(", ")}\n\nUsage: bun scripts/generate-against-server.ts --sample-count <n> --output-dir <dir> --concurrency <n> [--endpoint <url>]`,
  )
  process.exit(1)
}

const sampleCount = parsePositiveInt(requiredFlags["--sample-count"]!)
const concurrency = parsePositiveInt(requiredFlags["--concurrency"]!)
const outputDir = requiredFlags["--output-dir"]!

const endpointBase = args.endpoint ?? process.env.ENDPOINT_URL
if (!endpointBase) {
  console.error(
    "Missing endpoint URL. Provide --endpoint <url> or set ENDPOINT_URL in .env",
  )
  process.exit(1)
}
const endpointUrl = endpointBase.endsWith("/")
  ? `${endpointBase}generate`
  : `${endpointBase}/generate`

const connectionPairDir = join(outputDir, "images", "connection-pairs")
const routedDir = join(outputDir, "images", "routed")
await mkdir(connectionPairDir, { recursive: true })
await mkdir(routedDir, { recursive: true })

const startedAt = Date.now()
const rows: DatasetRow[] = []
const failures: Array<{ problemId: string; reason: string }> = []
let nextSampleIndex = 1
let cacheHits = 0
let cacheMisses = 0
let totalSolverAttempts = 0
let completed = 0

await Promise.all(
  Array.from({ length: Math.min(concurrency, sampleCount) }, () => worker()),
)

rows.sort((a, b) => a.id.localeCompare(b.id))
const jsonlPath = join(outputDir, "dataset.jsonl")
await writeFile(
  jsonlPath,
  `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
  "utf8",
)

const metadata = {
  created_at: new Date().toISOString(),
  endpoint: endpointUrl,
  requested_samples: sampleCount,
  generated_samples: rows.length,
  attempts: sampleCount,
  skipped: failures.length,
  concurrency,
  elapsed_ms: Date.now() - startedAt,
  cache_hits: cacheHits,
  cache_misses: cacheMisses,
  total_solver_attempts: totalSolverAttempts,
  average_solver_attempts:
    rows.length > 0 ? totalSolverAttempts / rows.length : 0,
  columns: [
    "boundary_connection_pairs",
    "connection_pair_image",
    "routed_image",
    "routed_paths",
  ],
}

await writeFile(
  join(outputDir, "metadata.json"),
  `${JSON.stringify(metadata, null, 2)}\n`,
  "utf8",
)

if (failures.length > 0) {
  await writeFile(
    join(outputDir, "failures.json"),
    `${JSON.stringify(failures, null, 2)}\n`,
    "utf8",
  )
}

if (rows.length < sampleCount) {
  console.warn(`generated ${rows.length}/${sampleCount} samples`)
}

console.log(`dataset written to ${jsonlPath}`)
console.log(
  `samples/minute: ${samplesPerMinute(completed, startedAt).toFixed(2)}`,
)
console.log(
  `cache hits: ${cacheHits}, cache misses: ${cacheMisses}, avg attempts: ${metadata.average_solver_attempts.toFixed(3)}`,
)

async function worker(): Promise<void> {
  while (true) {
    const index = nextSampleIndex
    nextSampleIndex += 1
    if (index > sampleCount) {
      return
    }

    const sampleId = `sample-${index.toString().padStart(6, "0")}`
    const seed = 1109 + index

    try {
      const response = await fetch(endpointUrl, {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          problemId: sampleId,
          seed,
          pairCount: pairCountForIndex(index),
          minPointSeparationMm: VIA_DIAMETER_MM,
          cellSizeMm: CELL_SIZE_MM,
          viaDiameterMm: VIA_DIAMETER_MM,
          traceThicknessMm: TRACE_THICKNESS_MM,
          traceMarginMm: TRACE_MARGIN_MM,
          maxSolveAttempts: MAX_SOLVE_ATTEMPTS,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const data = (await response.json()) as GeneratedSampleResponse
      if (!data.ok || !data.sample) {
        throw new Error(data.reason ?? "Server returned failure")
      }

      await writeSvgAndPng(
        join(connectionPairDir, `${sampleId}.svg`),
        join(connectionPairDir, `${sampleId}.png`),
        data.sample.connection_pair_svg,
      )
      await writeSvgAndPng(
        join(routedDir, `${sampleId}.svg`),
        join(routedDir, `${sampleId}.png`),
        data.sample.routed_svg,
      )

      rows.push({
        id: sampleId,
        boundary_connection_pairs: data.sample.boundary_connection_pairs,
        connection_pair_image: `images/connection-pairs/${sampleId}.png`,
        routed_image: `images/routed/${sampleId}.png`,
        routed_paths: data.sample.routed_paths,
      })

      totalSolverAttempts += data.attempts
      if (data.cached) {
        cacheHits += 1
      } else {
        cacheMisses += 1
      }

      completed += 1
      if (completed % 25 === 0 || completed === sampleCount) {
        console.log(
          `progress ${completed}/${sampleCount} (${samplesPerMinute(completed, startedAt).toFixed(2)} samples/min)`,
        )
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error"
      failures.push({ problemId: sampleId, reason: message })
      completed += 1
      console.warn(`skip ${sampleId}: ${message}`)
      if (completed % 25 === 0 || completed === sampleCount) {
        console.log(
          `progress ${completed}/${sampleCount} (${samplesPerMinute(completed, startedAt).toFixed(2)} samples/min)`,
        )
      }
    }
  }
}

async function writeSvgAndPng(
  svgPath: string,
  pngPath: string,
  svgContent: string,
): Promise<void> {
  await writeFile(svgPath, svgContent, "utf8")
  const pngBytes = new Resvg(svgContent, {
    fitTo: {
      mode: "width",
      value: IMAGE_SIZE_PX,
    },
  })
    .render()
    .asPng()
  await writeFile(pngPath, pngBytes)
}

function parsePositiveInt(raw: string): number {
  const value = Number(raw)
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Expected positive integer, got: ${raw}`)
  }

  return Math.floor(value)
}

function samplesPerMinute(completedCount: number, startedAtMs: number): number {
  const elapsedMinutes = Math.max((Date.now() - startedAtMs) / 60000, 1 / 60000)
  return completedCount / elapsedMinutes
}

function pairCountForIndex(index: number): number {
  const hash = ((index * 1103515245 + 12345) >>> 16) & 0x7fff
  return 2 + (hash % 9)
}
