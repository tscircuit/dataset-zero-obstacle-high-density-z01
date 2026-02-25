import { mkdir, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { Resvg } from "@resvg/resvg-js"

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

const sampleCountArg = process.argv[2]
const endpointArg = process.argv[3]
const outputDir = process.argv[4]
const concurrencyRaw = process.argv[5]

if (!sampleCountArg || !endpointArg || !outputDir || !concurrencyRaw) {
  throw new Error(
    "Usage: bun scripts/generate-against-server.ts <sampleCount> <endpointUrl> <outputDir> <concurrency>",
  )
}

const sampleCount = parsePositiveInt(sampleCountArg)
const concurrency = parsePositiveInt(concurrencyRaw)

const endpointUrl = endpointArg.endsWith("/")
  ? `${endpointArg}generate`
  : `${endpointArg}/generate`

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
          minPointSeparationMm: 0.61,
          cellSizeMm: 0.1,
          viaDiameterMm: 0.61,
          traceThicknessMm: 0.15,
          traceMarginMm: 0.1,
          maxSolveAttempts: 32,
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
      value: 1024,
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
