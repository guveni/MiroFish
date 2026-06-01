import { spawn, spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

function loadEnv(path) {
  if (!existsSync(path)) return {};
  const env = {};
  for (const rawLine of readFileSync(path, "utf8").split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = rawLine.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (!match) continue;
    env[match[1]] = match[2].replace(/^"|"$/g, "");
  }
  return env;
}

function repairEnvForDockerCompose(path) {
  if (!existsSync(path)) return;
  const input = readFileSync(path, "utf8");
  const lines = input.split("\n");
  let changed = false;
  const repaired = lines.map((rawLine) => {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) return rawLine;
    if (/^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=/.test(rawLine)) return rawLine;
    if (/^-{2,}.*-{2,}$/.test(line)) {
      changed = true;
      return `# ${line}`;
    }
    return rawLine;
  });
  if (changed) {
    writeFileSync(path, repaired.join("\n"));
    console.log("[dev] Repaired comment-only section headers in .env for Docker Compose.");
  }
}

function dockerContainerExists(name) {
  const result = spawnSync(
    "docker",
    ["ps", "-a", "--filter", `name=^/${name}$`, "--format", "{{.Names}}"],
    { encoding: "utf8" },
  );
  return result.status === 0 && result.stdout.trim() === name;
}

function waitForNeo4jContainer(name) {
  for (let i = 0; i < 60; i += 1) {
    const status = spawnSync(
      "docker",
      ["inspect", "--format={{.State.Health.Status}}", name],
      { encoding: "utf8" },
    );
    if (status.status === 0 && status.stdout.trim() === "healthy") return true;
    const running = spawnSync(
      "docker",
      ["inspect", "--format={{.State.Running}}", name],
      { encoding: "utf8" },
    );
    if (running.status === 0 && running.stdout.trim() === "true" && status.stdout.trim() === "") {
      return true;
    }
    spawnSync("sleep", ["1"], { stdio: "ignore" });
  }
  return false;
}

const env = { ...loadEnv(".env"), ...process.env };
const backend = (env.GRAPH_BACKEND || "neo4j").trim().toLowerCase();

if (backend === "neo4j" || backend === "graphiti") {
  console.log("[dev] Neo4j backend (default) - ensuring Neo4j is up...");
  repairEnvForDockerCompose(".env");
  if (dockerContainerExists("mirofish-neo4j")) {
    const started = spawnSync("docker", ["start", "mirofish-neo4j"], {
      stdio: "inherit",
      env,
    });
    if (started.status !== 0 || !waitForNeo4jContainer("mirofish-neo4j")) {
      console.error("[dev] Existing mirofish-neo4j container did not become healthy.");
      process.exit(started.status ?? 1);
    }
  } else {
    const result = spawnSync("npm", ["run", "graph:up"], {
      stdio: "inherit",
      env,
    });
    if (result.status !== 0) {
      console.error("[dev] Failed to start Neo4j. Run `npm run graph:logs` for details.");
      process.exit(result.status ?? 1);
    }
  }
} else {
  console.log(`[dev] GRAPH_BACKEND=${backend} - skipping Neo4j container.`);
}

function ensureInngestContainer() {
  console.log("[dev] Ensuring Inngest Dev Server is up...");
  const backendPort = env.FLASK_PORT || "5001";
  if (dockerContainerExists("mirofish-inngest")) {
    spawnSync("docker", ["start", "mirofish-inngest"], {
      stdio: "inherit",
    });
  } else {
    spawnSync(
      "docker",
      [
        "run",
        "-d",
        "--name",
        "mirofish-inngest",
        "-p",
        "8288:8288",
        "--add-host=host.docker.internal:host-gateway",
        "inngest/inngest:latest",
        "inngest",
        "dev",
        "-u",
        `http://host.docker.internal:${backendPort}/api/inngest`,
        "--host",
        "0.0.0.0",
      ],
      { stdio: "inherit" }
    );
  }
}

ensureInngestContainer();

const child = spawn(
  "concurrently",
  [
    "--kill-others",
    "-n",
    "backend,frontend",
    "-c",
    "green,cyan",
    "npm run backend",
    "npm run frontend",
  ],
  { stdio: "inherit", env },
);

const forward = (signal) => {
  try {
    child.kill(signal);
  } catch {}
};

process.on("SIGINT", () => forward("SIGINT"));
process.on("SIGTERM", () => forward("SIGTERM"));
child.on("exit", (code) => process.exit(code ?? 0));
