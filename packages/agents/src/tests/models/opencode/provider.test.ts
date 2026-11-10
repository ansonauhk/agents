import { describe, expect, it } from "vitest";
import { createOpenAI } from "@ai-sdk/openai";
import type { LanguageModelV3 } from "@ai-sdk/provider";
import type { Plugin } from "@opencode/plugin";
import { CLOUDFLARE_PROVIDER_ID, createAI } from "../../../models/opencode";
import { asAi, fakeGatewayBinding, jsonResponse } from "../ai-sdk/helpers";

const binding = { aiGatewayLogId: null } as unknown as Ai;

describe("agents/models/opencode", () => {
  it("registers a vendor model and resolves its native gateway model in the language hook", async () => {
    const binding = fakeGatewayBinding({
      native: () =>
        jsonResponse({
          id: "resp_test",
          object: "response",
          created_at: 1,
          model: "my-deployment",
          status: "completed",
          output: [
            {
              type: "message",
              id: "msg_test",
              role: "assistant",
              status: "completed",
              content: [{ type: "output_text", text: "OK", annotations: [] }]
            }
          ],
          usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 }
        })
    });
    const ai = createAI({ binding: asAi(binding), id: "azure-test" });
    const azure = createOpenAI({
      apiKey: "placeholder",
      baseURL: "https://acme.openai.azure.com/openai/v1"
    });
    const ref = ai(azure.responses("my-deployment"), {
      limit: { context: 128_000, output: 16 }
    });
    expect(ref).toEqual({
      providerID: CLOUDFLARE_PROVIDER_ID,
      id: "openai.responses/my-deployment"
    });

    type LanguageEvent = { model: { id: string }; language?: LanguageModelV3 };
    let languageHook: ((event: LanguageEvent) => void) | undefined;
    let registered: readonly {
      id: string;
      limit: { context: number; output: number };
    }[] = [];
    const context = {
      provider: {
        async transform(
          callback: (editor: {
            add(input: { models: typeof registered }): void;
          }) => void
        ) {
          callback({
            add: (input) => {
              registered = input.models;
            }
          });
          return { dispose: async () => {} };
        },
        reload: async () => {}
      },
      aisdk: {
        async hook(_name: string, handler: (event: LanguageEvent) => void) {
          languageHook = handler;
          return { dispose: async () => {} };
        }
      }
    } as unknown as Plugin.Context;
    const cleanup = await ai.plugin.setup(context);
    expect(registered).toContainEqual(
      expect.objectContaining({
        id: ref.id,
        limit: { context: 128_000, output: 16 }
      })
    );
    const event: LanguageEvent = { model: { id: ref.id } };
    languageHook!(event);
    const result = await event.language!.doGenerate({
      prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
      maxOutputTokens: 16
    });
    expect(result.content).toEqual([
      {
        type: "text",
        text: "OK",
        providerMetadata: { openai: { itemId: "msg_test" } }
      }
    ]);
    expect(binding.native[0]).toMatchObject({
      provider: "azure-openai",
      endpoint: "acme/openai/responses",
      gatewayId: "azure-test",
      query: { model: "my-deployment", max_output_tokens: 16 }
    });
    if (cleanup) await cleanup();
  });

  it("takes Workers AI ids, catalog ids and dynamic routes", () => {
    const ai = createAI({ binding });
    for (const id of [
      "@cf/moonshotai/kimi-k2.7-code",
      "anthropic/claude-opus-4.8",
      "dynamic/support"
    ] as const) {
      expect(ai(id)).toEqual({ id, providerID: CLOUDFLARE_PROVIDER_ID });
    }
  });

  it("refuses an id that names no model where it is named", () => {
    const ai = createAI({ binding });
    expect(() => ai("claude-opus-4-8" as never)).toThrow(
      /not a model id this provider knows/
    );
  });
});
