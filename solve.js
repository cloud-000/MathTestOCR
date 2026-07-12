#!/usr/bin/env node
/*
 * solve.js — generate worked solutions for parsed tests via a free OpenRouter model.
 *
 * Some series (older MathCounts rounds especially) ship no solutions. This tool
 * reads a test's `problems.json`, asks a (free) OpenRouter model to solve each
 * problem, and writes `problem_solution.json` in the same on-disk schema the
 * Python pipeline uses: { "<n>": ["<solution string>"] }.
 *
 * Design notes:
 *   - Zero dependencies. Node 18+ (global fetch). Reads OPENROUTER_API_KEY from
 *     the environment or a local `.env` file.
 *   - Grounded solving: when a known answer exists (problem_answer.json), it is
 *     given to the model so it produces a *correct* worked solution rather than
 *     guessing. With no key, it solves blind.
 *   - Resumable + crash-safe: already-solved problems are skipped, and
 *     problem_solution.json is rewritten atomically (tmp + rename) after each
 *     completion, so an interrupted run loses nothing.
 *   - Rate-limit aware: honors 429 / Retry-After with exponential backoff+jitter
 *     and falls through an ordered list of models.
 *
 * Usage:
 *   node solve.js --series mathcounts --test 2000_national_cdr
 *   node solve.js --series mathcounts --all
 *   node solve.js --series mathcounts --test 2000_national_cdr \
 *       --model deepseek/deepseek-r1:free --concurrency 3 --force
 *
 * Flags:
 *   --series <name>     series folder under out/ (default: mathcounts)
 *   --test <id>         single test id (folder under out/<series>/)
 *   --all               every test in the series that has problems.json
 *   --out <dir>         output root (default: out)
 *   --effort <tier>     model tier: easy | medium | hard (default: medium)
 *   --model <a,b,...>   override the effort tier with an explicit fallback list
 *   --concurrency <n>   parallel requests (default: 3 — free tiers are strict)
 *   --limit <n>         only solve the first n unsolved problems (debugging)
 *   --force             re-solve problems that already have a solution
 *   --dry-run           print the plan, make no API calls, write nothing
 */

"use strict";

const fs = require("fs");
const fsp = require("fs/promises");
const path = require("path");

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

// Free-model fallback lists by effort. --effort selects one; entries are tried
// in order (fallback chain). Override any tier wholesale with --model a,b,c.
const MODEL_TIERS = {
    easy: [
        "poolside/laguna-xs-2.1:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ],
    medium: [
        "meta-llama/llama-3.3-70b-instruct:free",
        "openai/gpt-oss-120b:free",
        "poolside/laguna-m.1:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ],
    hard: [
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "tencent/hy3:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    ],
};

const DEFAULT_EFFORT = "medium";

const DEFAULTS = {
    series: "mathcounts",
    out: "out",
    concurrency: 3,
    maxRetries: 5,
    temperature: 0.2,
    // Optional attribution headers OpenRouter recommends (harmless if unused).
    referer: "https://github.com/local/comp-OCR",
    title: "comp-OCR solve.js",
};

const IMAGE_RE = /!\[[^\]]*\]\([^)]*\)/g; // markdown image marker in a statement

// ---------------------------------------------------------------------------
// Tiny .env loader (no dependency). Does not override already-set env vars.
// ---------------------------------------------------------------------------

function loadDotEnv(file) {
    let text;
    try {
        text = fs.readFileSync(file, "utf8");
    } catch {
        return;
    }
    for (const raw of text.split("\n")) {
        const line = raw.trim();
        if (!line || line.startsWith("#")) continue;
        const eq = line.indexOf("=");
        if (eq === -1) continue;
        const key = line.slice(0, eq).trim();
        let val = line.slice(eq + 1).trim();
        if (
            (val.startsWith('"') && val.endsWith('"')) ||
            (val.startsWith("'") && val.endsWith("'"))
        ) {
            val = val.slice(1, -1);
        }
        if (!(key in process.env)) process.env[key] = val;
    }
}

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------

function parseArgs(argv) {
    const opts = {
        series: DEFAULTS.series,
        out: DEFAULTS.out,
        test: null,
        all: false,
        effort: DEFAULT_EFFORT,
        modelOverride: null,
        concurrency: DEFAULTS.concurrency,
        limit: Infinity,
        force: false,
        dryRun: false,
    };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        const next = () => argv[++i];
        switch (a) {
            case "--series":
                opts.series = next();
                break;
            case "--test":
                opts.test = next();
                break;
            case "--all":
                opts.all = true;
                break;
            case "--out":
                opts.out = next();
                break;
            case "--effort": {
                const e = next();
                if (!(e in MODEL_TIERS)) {
                    console.error(
                        `Invalid --effort "${e}" (expected: ${Object.keys(MODEL_TIERS).join(", ")})`,
                    );
                    opts.help = true;
                } else {
                    opts.effort = e;
                }
                break;
            }
            case "--model":
                opts.modelOverride = next()
                    .split(",")
                    .map((s) => s.trim())
                    .filter(Boolean);
                break;
            case "--concurrency":
                opts.concurrency = Math.max(1, parseInt(next(), 10) || 1);
                break;
            case "--limit":
                opts.limit = Math.max(0, parseInt(next(), 10) || 0);
                break;
            case "--force":
                opts.force = true;
                break;
            case "--dry-run":
                opts.dryRun = true;
                break;
            case "-h":
            case "--help":
                opts.help = true;
                break;
            default:
                console.error(`Unknown argument: ${a}`);
                opts.help = true;
        }
    }
    // --model overrides the effort tier wholesale; otherwise use the tier list.
    opts.models = opts.modelOverride || MODEL_TIERS[opts.effort];
    return opts;
}

// ---------------------------------------------------------------------------
// Disk I/O
// ---------------------------------------------------------------------------

async function readJson(file, fallback) {
    try {
        return JSON.parse(await fsp.readFile(file, "utf8"));
    } catch (e) {
        if (e.code === "ENOENT") return fallback;
        throw new Error(`Failed to parse ${file}: ${e.message}`);
    }
}

// Atomic write: tmp file in the same dir + rename (rename is atomic on same fs).
async function writeJsonAtomic(file, obj) {
    const tmp = `${file}.tmp`;
    await fsp.writeFile(tmp, JSON.stringify(obj, null, 2) + "\n");
    await fsp.rename(tmp, file);
}

async function discoverTests(seriesDir) {
    const entries = await fsp.readdir(seriesDir, { withFileTypes: true });
    const tests = [];
    for (const e of entries) {
        if (!e.isDirectory() || e.name.startsWith("_")) continue;
        if (fs.existsSync(path.join(seriesDir, e.name, "problems.json"))) {
            tests.push(e.name);
        }
    }
    return tests.sort();
}

// ---------------------------------------------------------------------------
// Prompt construction + answer extraction
// ---------------------------------------------------------------------------

function buildMessages(statement, knownAnswer) {
    const hasImage = IMAGE_RE.test(statement);
    const cleanStatement = statement
        .replace(IMAGE_RE, "[figure omitted]")
        .trim();

    const system =
        "You are an expert competition-math solver (MathCounts / AMC level). " +
        "Write a clear, rigorous, concise worked solution in Markdown with LaTeX " +
        "(inline $...$, display $$...$$). Do not restate the problem. End with the " +
        "final answer on its own line as \\boxed{...}.";

    let user = `Problem:\n${cleanStatement}\n\n`;
    if (hasImage) {
        user +=
            "NOTE: this problem references a figure that is not available. Infer the " +
            "most reasonable configuration from the text and state any assumption.\n\n";
    }
    if (knownAnswer != null && String(knownAnswer).trim() !== "") {
        user +=
            `The correct final answer is: ${knownAnswer}\n` +
            "Produce a solution that rigorously arrives at exactly this answer, and " +
            "end with it in \\boxed{...}.";
    } else {
        user += "Solve it and end with the final answer in \\boxed{...}.";
    }

    return [
        { role: "system", content: system },
        { role: "user", content: user },
    ];
}

function extractBoxed(text) {
    // Return the content of the last \boxed{...}, matching balanced braces.
    const marker = "\\boxed{";
    let last = null;
    let idx = text.indexOf(marker);
    while (idx !== -1) {
        let depth = 1;
        let i = idx + marker.length;
        for (; i < text.length && depth > 0; i++) {
            if (text[i] === "{") depth++;
            else if (text[i] === "}") depth--;
        }
        if (depth === 0) last = text.slice(idx + marker.length, i - 1).trim();
        idx = text.indexOf(marker, idx + marker.length);
    }
    return last;
}

// Loose answer comparison for the verification report.
function answersMatch(a, b) {
    const norm = (s) =>
        String(s)
            .toLowerCase()
            .replace(/\\[a-z]+/g, "") // latex commands
            .replace(/[\s${}]/g, "") // whitespace, $, braces
            .replace(/[^a-z0-9./-]/g, ""); // keep digits, letters, a few math chars
    return norm(a) === norm(b);
}

// ---------------------------------------------------------------------------
// OpenRouter client
// ---------------------------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function backoffDelay(attempt, retryAfterHeader) {
    if (retryAfterHeader) {
        const secs = parseInt(retryAfterHeader, 10);
        if (!Number.isNaN(secs)) return secs * 1000;
    }
    const base = Math.min(30000, 1000 * 2 ** attempt);
    return base + Math.floor(Math.random() * 500); // jitter
}

async function chat(apiKey, models, messages, opts) {
    let lastErr;
    for (const model of models) {
        for (let attempt = 0; attempt <= DEFAULTS.maxRetries; attempt++) {
            let res;
            try {
                res = await fetch(OPENROUTER_URL, {
                    method: "POST",
                    headers: {
                        Authorization: `Bearer ${apiKey}`,
                        "Content-Type": "application/json",
                        "HTTP-Referer": DEFAULTS.referer,
                        "X-Title": DEFAULTS.title,
                    },
                    body: JSON.stringify({
                        model,
                        messages,
                        temperature: DEFAULTS.temperature,
                    }),
                });
            } catch (e) {
                lastErr = e; // network error — retry
                await sleep(backoffDelay(attempt));
                continue;
            }

            if (res.status === 429 || res.status >= 500) {
                lastErr = new Error(`${model}: HTTP ${res.status}`);
                if (attempt < DEFAULTS.maxRetries) {
                    await sleep(
                        backoffDelay(attempt, res.headers.get("retry-after")),
                    );
                    continue;
                }
                break; // exhausted retries for this model → try next model
            }

            if (!res.ok) {
                const body = await res.text().catch(() => "");
                lastErr = new Error(
                    `${model}: HTTP ${res.status} ${body.slice(0, 300)}`,
                );
                break; // 4xx (bad request/auth/model gone) → try next model, no retry
            }

            const data = await res.json();
            const content = data?.choices?.[0]?.message?.content;
            if (!content) {
                lastErr = new Error(`${model}: empty completion`);
                break;
            }
            return { content, model };
        }
    }
    throw lastErr || new Error("All models failed");
}

// ---------------------------------------------------------------------------
// Concurrency pool
// ---------------------------------------------------------------------------

async function runPool(items, concurrency, worker) {
    let cursor = 0;
    const runners = Array.from(
        { length: Math.min(concurrency, items.length) },
        async () => {
            while (cursor < items.length) {
                const idx = cursor++;
                await worker(items[idx], idx);
            }
        },
    );
    await Promise.all(runners);
}

// ---------------------------------------------------------------------------
// Per-test solving
// ---------------------------------------------------------------------------

async function solveTest(testDir, apiKey, opts) {
    const problems = await readJson(path.join(testDir, "problems.json"), null);
    if (!problems) {
        console.warn(`  skip: no problems.json in ${testDir}`);
        return;
    }
    const answers = await readJson(
        path.join(testDir, "problem_answer.json"),
        {},
    );
    const solPath = path.join(testDir, "problem_solution.json");
    const answersPath = path.join(testDir, "problem_answer.json");
    const solutions = opts.force ? {} : await readJson(solPath, {});

    // Preserve the problem order from problems.json.
    const numbers = Object.keys(problems);
    let todo = numbers.filter((n) => opts.force || !solutions[n]);
    if (Number.isFinite(opts.limit)) todo = todo.slice(0, opts.limit);

    const label = path.basename(testDir);
    if (todo.length === 0) {
        console.log(
            `  ${label}: nothing to do (${numbers.length} problems already solved)`,
        );
        return;
    }
    console.log(
        `  ${label}: solving ${todo.length}/${numbers.length} ` +
            `(${Object.keys(answers).length} answers known)`,
    );

    if (opts.dryRun) return;

    let done = 0;
    let derivedAnswers = 0;
    const mismatches = [];
    // Serialize disk writes so concurrent workers never interleave a rewrite.
    let writeChain = Promise.resolve();
    const persist = (writeAnswers) => {
        writeChain = writeChain
            .then(() => writeJsonAtomic(solPath, solutions))
            .then(() =>
                writeAnswers ? writeJsonAtomic(answersPath, answers) : null,
            );
        return writeChain;
    };

    await runPool(todo, opts.concurrency, async (n) => {
        const knownAnswer = answers[n];
        const messages = buildMessages(problems[n], knownAnswer);
        try {
            const { content, model } = await chat(
                apiKey,
                opts.models,
                messages,
                opts,
            );
            solutions[n] = [content.trim()];

            const got = extractBoxed(content);
            // No answer key for this problem: adopt the model's boxed answer.
            const derived = knownAnswer == null && got != null;
            if (derived) {
                answers[n] = got;
                derivedAnswers++;
            }
            await persist(derived);

            const mismatch =
                knownAnswer != null &&
                got != null &&
                !answersMatch(got, knownAnswer);
            if (mismatch) mismatches.push({ n, expected: knownAnswer, got });
            done++;
            process.stdout.write(
                `    [${done}/${todo.length}] #${n} ok (${model})` +
                    (derived ? ` → answer ${got}` : "") +
                    (mismatch ? " ⚠ answer mismatch" : "") +
                    "\n",
            );
        } catch (e) {
            done++;
            console.error(
                `    [${done}/${todo.length}] #${n} FAILED: ${e.message}`,
            );
        }
    });

    await writeChain; // flush final write
    console.log(`  ${label}: wrote ${solPath}`);
    if (derivedAnswers) {
        console.log(
            `  ${label}: derived ${derivedAnswers} answer(s) → ${answersPath}`,
        );
    }
    if (mismatches.length) {
        console.warn(`  ${label}: ${mismatches.length} answer mismatch(es):`);
        for (const m of mismatches) {
            console.warn(
                `     #${m.n}: expected ${m.expected}, model said ${m.got}`,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const HELP = `solve.js — generate solutions via OpenRouter

  node solve.js --series mathcounts --test 2000_national_cdr
  node solve.js --series mathcounts --all
  node solve.js --series mathcounts --all --effort hard
  node solve.js --series mathcounts --test <id> --model deepseek/deepseek-r1:free --force

Flags: --series --test --all --out --effort(easy|medium|hard) --model --concurrency --limit --force --dry-run
Needs OPENROUTER_API_KEY (env or .env file).`;

async function main() {
    loadDotEnv(path.join(process.cwd(), ".env"));
    const opts = parseArgs(process.argv.slice(2));
    if (opts.help) {
        console.log(HELP);
        return;
    }

    const seriesDir = path.join(opts.out, opts.series);
    if (!fs.existsSync(seriesDir)) {
        console.error(`No such series directory: ${seriesDir}`);
        process.exitCode = 1;
        return;
    }

    let tests;
    if (opts.all) {
        tests = await discoverTests(seriesDir);
    } else if (opts.test) {
        tests = [opts.test];
    } else {
        console.error("Specify --test <id> or --all.");
        process.exitCode = 1;
        return;
    }

    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey && !opts.dryRun) {
        console.error("OPENROUTER_API_KEY is not set (env or .env file).");
        process.exitCode = 1;
        return;
    }

    const modelDesc = opts.modelOverride ? "override" : `effort=${opts.effort}`;
    console.log(
        `series=${opts.series} tests=${tests.length} ${modelDesc} ` +
            `models=[${opts.models.join(", ")}] ` +
            `concurrency=${opts.concurrency}${opts.dryRun ? " (dry-run)" : ""}`,
    );

    for (const t of tests) {
        const dir = path.join(seriesDir, t);
        if (!fs.existsSync(dir)) {
            console.warn(`skip: ${dir} does not exist`);
            continue;
        }
        console.log(`\n${t}`);
        await solveTest(dir, apiKey, opts);
    }
}

main().catch((e) => {
    console.error(e);
    process.exitCode = 1;
});
