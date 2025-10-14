import { z } from "zod";
import chalk from "chalk";
import { Sema } from "async-sema";
import ky from "ky";

import * as path from "node:path";
import * as fsPromise from "node:fs/promises";

const APP_ID = process.env.APP_ID;
const API_TOKEN = process.env.API_TOKEN;

if (
  typeof APP_ID !== "string" ||
  typeof API_TOKEN !== "string" ||
  !APP_ID ||
  !API_TOKEN
) {
  console.error(
    chalk.redBright(chalk.bold("ERROR")),
    "Missing `APP_ID` or `API_TOKEN`, did you forget to add environment variables?"
  );
  process.exit(1);
}

const QuestionSchema = z.object({
  pid: z.string(),
  content: z.string(),
});

const ResultSchema = z.object({
  data: z.object({
    judge: z.object({
      score: z.int(),
    }),
  }),
  requestId: z.string(),
  trackId: z.string(),
});

const category = ["normal", "thinking"] as const;

const luoguWs = new WebSocket(
  `wss://open-ws.lgapi.cn/ws?token=${API_TOKEN}&channel=${encodeURIComponent(
    "judge.result"
  )}`
);

for (const suffix of category) {
  const directory = path.resolve(
    import.meta.dirname,
    "..",
    `qwen3_30B_${suffix}`
  );

  const tests = (await fsPromise.readdir(directory)).filter((v) =>
    v.startsWith("e")
  );

  for (const test of tests) {
    const filePath = path.resolve(directory, test);

    const file = await fsPromise.readFile(filePath, "utf-8");

    const sema = new Sema(5);

    const scorePath = path.resolve(directory, test.replace(/^e/, "result"));

    await fsPromise.writeFile(scorePath, "");

    console.log(
      chalk.yellowBright(chalk.bold("WARN")),
      `Processing ${filePath}, results are writing into ${scorePath}...`
    );

    await Promise.all(
      file
        .trim()
        .split("\n")
        .map((line) => {
          try {
            return QuestionSchema.parse(JSON.parse(line));
          } catch (err) {
            console.error(
              chalk.redBright(chalk.bold("ERROR")),
              `Parse JSON failed: ${err}`
            );

            process.exit(1);
          }
        })
        .map(async ({ pid, content }) => {
          await sema.acquire();

          const uid = crypto.randomUUID();

          console.log(chalk.greenBright("INFO"), `Starting on ${pid}`);

          const code = content
            .match(/\`\`\`\n?(?:cpp)?\n?([^`]+)(?:\`\`\`|$)/s)?.[1]
            ?.trim();

          if (!code) {
            await fsPromise.appendFile(
              scorePath,
              `${JSON.stringify({ pid, score: 0 })}\n`,
              "utf-8"
            );

            console.log(
              chalk.yellowBright(chalk.bold("WARN")),
              `Reading empty code, ${pid} finished`
            );

            sema.release();

            return;
          }

          const handleMessage = (ev: WebSocketEventMap["message"]) => {
            if (typeof ev.data === "string") {
              const [channel, payload] = z
                .tuple([z.string(), z.string()])
                .parse(ev.data.split("\x00"));

              if (channel === "judge.result") {
                const payloadJson = ResultSchema.parse(JSON.parse(payload));

                if (payloadJson.trackId === uid) {
                  fsPromise
                    .appendFile(
                      scorePath,
                      `${JSON.stringify({
                        pid,
                        score: payloadJson.data.judge.score,
                      })}\n`,
                      "utf-8"
                    )
                    .then(() => {
                      console.log(chalk.greenBright("INFO"), `${pid} finished`);
                    })
                    .finally(() => {
                      luoguWs.removeEventListener("message", handleMessage);
                      sema.release();
                    });
                }
              } else if (channel === "AUTH_WELCOME") {
                console.log(chalk.cyan("DEBUG"), "WebSocket connected");
              }
            }
          };

          luoguWs.addEventListener("message", handleMessage);

          const response = await ky.post(
            "https://open-v1.lgapi.cn/judge/problem",
            {
              headers: {
                Authorization: `Basic ${Buffer.from(API_TOKEN).toString(
                  "base64"
                )}`,
              },
              json: {
                pid,
                lang: "cxx/14/gcc",
                o2: true,
                code,
                trackId: uid,
              },
              throwHttpErrors: false,
            }
          );

          if (response.status > 399) {
            console.error(
              chalk.redBright(chalk.bold("ERROR")),
              `Request judge failed: ${response.statusText}: ${
                ((await response.json()) as any).errorMessage
              }`
            );

            luoguWs.removeEventListener("message", handleMessage);
            sema.release();

            if (response.status === 400) {
              console.error(
                chalk.redBright(chalk.bold("ERROR")),
                `Skipping 400...`
              );
            } else {
              process.exit(1);
            }
          }
        })
    );

    await sema.drain();

    console.log(chalk.yellowBright(chalk.bold("WARN")), `Finish ${filePath}`);
  }
}
