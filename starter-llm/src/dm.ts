import { assign, createActor, fromPromise, setup } from "xstate";
import { Settings, speechstate } from "speechstate";
import { PROXY_KEY, NLU_KEY } from "./credentials";
import { DMContext, DMEvents, Message, NLUObject, ExtractedRequest, DialogueIntent } from "./types";
import OpenAI from "openai";

const REGION = "northeurope";

const openai = new OpenAI({
  baseURL: "http://localhost:11434/v1/",
  apiKey: "ollama",
  dangerouslyAllowBrowser: true,
});

// Azure access via FLoV proxy
const azureProxyCredentials = {
  proxyUrl: "https://rndserv.flov.gu.se:4000/api/token",
  key: PROXY_KEY,
};

// NLU access-> needed for intent
const azureLanguageCredentials = {
  endpoint: "https://lab-gusbeihel.cognitiveservices.azure.com/language/:analyze-conversations?api-version=2024-11-15-preview",
  key: NLU_KEY, 
  deploymentName: "IdiomExplainer",
  projectName: "idiom_explainer",
  };

const settings: Settings = {
  azureLanguageCredentials: azureLanguageCredentials,
  azureCredentials: azureProxyCredentials,
  azureRegion: REGION,
  asrDefaultCompleteTimeout: 0,
  asrDefaultNoInputTimeout: 5000,
  locale: "en-US",
  ttsDefaultVoice: "en-US-DavisNeural",
  bargeIn: false, // ENABLE ????
};

// checking the confidence score for the intents
function getIntentConfidence(nlu: NLUObject | null, intent: string): number {
  return nlu?.intents.find(i => i.category === intent)?.confidenceScore ?? 0;
};

const dmMachine = setup({
  types: {
    context: {} as DMContext,
    events: {} as DMEvents,
  },
  actions: {
    "spst.speak": ({ context }, params: { utterance: string }) =>
      context.spstRef.send({
        type: "SPEAK",
        value: {
          utterance: params.utterance,
        },
      }),
    "spst.listen": ({ context }) =>
      context.spstRef.send({
        type: "LISTEN",
        value: { nlu: true }, // local activation of NLU
      }),
  },
  actors: {

    // generation step: current dialogue history -> LLM + returns text response
    fetchLLM: fromPromise<string, { messages: Message[] }>(
      async ({ input }) => {
        //console.log("Actor input:", input);

        const response = await openai.chat.completions.create({
          model: "llama3.2:latest",
          messages: input.messages,
        });

      //console.log("LLM response:", response);

      return response.choices[0].message.content ?? "";
      },
    ),

    // using the LLM to extract idiom and context from user utterance
    extractRequest: fromPromise<ExtractedRequest, { messages: Message[] }>(
      async ({ input }) => {

      const response = await openai.chat.completions.create({
        model: "llama3.2:latest",
        messages: input.messages,
        response_format: { type: "json_object" },
      });

      const content = response.choices[0].message.content ?? "{}";

      console.log("LLM response:", content);

      return JSON.parse(content) as ExtractedRequest;
    }),

  },
}).createMachine({
  context: ({ spawn }) => ({
    spstRef: spawn(speechstate, { input: settings }),

    // adding interpretation (for intents)
    interpretation: null as NLUObject | null,

    // for storing ASR results for current turn -> resets before going back to listening
    lastResult: null,

    // for storing idiom under discussion
    currentIdiom: "",

    // for storing context provided by user
    userContext: "",

    // used to distiguish tasks in LLM prompt
    dialogueIntent: "explain_idiom" as DialogueIntent,

    // for storing LLM generated examples
    generatedExample: "",

    // has the system already provided an explanation?
    hasExplainedIdiom: false,

    // dialogue history across multiple turns
    messages: [
      {
        role: "system",
        content:
          "You are a helpful idiom explainer for English language learners.",
      },
    ] as Message[],

    noInputCount: 0, // counting how often there is no input from the user
  }),
  id: "DM",
  initial: "Prepare",
  states: {
    Prepare: {
      entry: ({ context }) => context.spstRef.send({ type: "PREPARE" }),
      on: { ASRTTS_READY: "WaitToStart" },
    }, // Prepare end
    WaitToStart: {
      // waiting until the button is clicked

      // resetting dialogue context for new conversation
      entry: assign({
        currentIdiom: "",
        userContext: "",
        generatedExample: "",
        dialogueIntent: "explain_idiom",
        hasExplainedIdiom: false,

        interpretation: null,
        lastResult: null,
        noInputCount: 0,

        messages: [
          {
            role: "system",
            content:
              "You are a helpful idiom explainer for English language learners.",
          },
        ],
      }),

      on: { 
        CLICK: {
          target: "Greeting",
        },
      },
    }, // WaitToStart end
    Greeting: {
      // greeting + initial user ask is analyzed
      initial: "Prompt",
      on: {
        LISTEN_COMPLETE: [
          
          // detecting intent (NLU) + recognizing utterance

          // user asks about new idiom
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "new_idiom" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "new_idiom") > 0.65,

            // setting dialogue intent  
            actions: assign({
              dialogueIntent: "explain_idiom",

            // keeping context if no idiom stored (context provided first)
            // clearing old context if idiom already stored (new idiom provided)
              userContext: ({ context }) =>
                context.currentIdiom ? "" : context.userContext,
            }),

            target: "ExtractRequest" // ???
          },
          // user provides context
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "user_context" &&
              getIntentConfidence(context.interpretation, "user_context") > 0.65,

            actions: assign({
              userContext: ({ context }) =>
                context.lastResult?.[0]?.utterance ?? "",

              dialogueIntent: "explain_context",
            }),

            target: "CheckIdiom" // ???
          },
          // user wants the system to provide a context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "request_example" &&
              getIntentConfidence(context.interpretation, "request_example") > 0.65,
            
            actions: assign({
              dialogueIntent: "request_example",
            }),

            target: "CheckIdiom",
          },
          // user wants to hear different context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "different_example" &&
              getIntentConfidence(context.interpretation, "different_example") > 0.65,
            
            actions: assign({
              dialogueIntent: "request_example",
            }),

            target: "CheckIdiom",
          },
          // user wants more in depth explanation of the current idiom
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "more_details" &&
              getIntentConfidence(context.interpretation, "more_details") > 0.65,
            
            actions: assign({
              dialogueIntent: "more_details",
            }),

            target: "CheckIdiom",
          },

          // user says no (from AnythingMore)
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "negation" &&
              getIntentConfidence(context.interpretation, "negation") > 0.65,

            target: "Done"
          },

          // handling unclear input
          { target: "Clarify",
            guard: ({ context }) => !!context.lastResult, // CHECK !!!!
          },

          // third time no input (consecutively)
          { target: ".NoInputFinal",
            guard: ({ context }) => context.noInputCount >= 3,
          },
          // first or second time no input
          {
            target: ".NoInput",
          },
        ],
      }, // on: LISTEN_COMPLETE end
      states: {
        Prompt: { // greeting the user
          entry: { type: "spst.speak", params: { utterance: `Welcome to Idiomagic - your personal idiom explainer. Which idiom is on your mind today?` } },
          on: { SPEAK_COMPLETE: "Ask" },
        }, // Prompt end
        NoInput: { // handling first and second time no input
          entry: {
            type: "spst.speak",
            params: ({ context }) => ({
                utterance:
                  context.noInputCount === 1
                    ? `I didn't hear anything. Please try again.` // if count === 1
                    : `I still didn't hear anything. Please try one more time`, // if count === 2 (ALSO IF 0 !!!)
            }),
          },
          on: { SPEAK_COMPLETE: "Ask" },
        }, // NoInput end
        NoInputFinal: { // handling third time no input
          entry: {
            type: "spst.speak",
            params: { utterance: `I didn't hear anything. Click to try again! ` }, 
          },
          on: { SPEAK_COMPLETE: "#DM.WaitToStart" }, // sending the user back to the beginning after three times of no input
        }, // NoInputFinal end
        Ask: { // handling content of user ask
          entry: [
            assign({ 
              lastResult: null, // resetting lastResult
              interpretation: null, // resetting intent
            }), 
            { type: "spst.listen" },
          ],
          on: {
            RECOGNISED: {
              actions: [
                assign({interpretation: ({ event }) => event.nluValue ?? null}), // extracting intent
              
                assign(({ context, event }) => {
                  const utterance = event.value[0].utterance; // extracting what the user said

                  return {
                    lastResult: event.value,
                    messages: [
                      ...context.messages, // appending to convo history
                      {
                        role: "user" as const, // as user message
                        content: utterance,
                      },
                    ],
                    noInputCount: 0, // resetting no input counter if utterance recognised
                  };
                }),
              ],
            }, // RECOGNISED end
            ASR_NOINPUT: {

              // handling no input counter
              actions:
                assign(({ context }) => ({
                    lastResult: null, // no Input -> clear lastResult
                    interpretation: null, // clear interpretation !!!!
                    noInputCount: context.noInputCount + 1, // increasing the count by one
              })),
            }, // ASR_NOINPUT end
          }, // on: end
        }, // Ask end
      },
    }, // Greeting end
    Clarify: {
      // the system did not understand / the user utterance did not match any intent
      // the system asks the user to repeat their ask

      entry: {
        type: "spst.speak",
        params: {
          utterance:
            "Sorry, I didn't understand. Could you rephrase that?",
        },
      },
      on: {
        SPEAK_COMPLETE: "Greeting.Ask", // change to whatever state it came from !!!!
      },
    }, // Clarify end
    ExtractRequest: {
      // sending the user utterance to LLM to extract the new idiom and possibly the context

      // prompting LLM
      invoke: {
        src: "extractRequest",
        input: ({ context }) => ({
          messages: [
            {
              role: "system" as const,
              content: `
                Extract the idiom and any situation or context provided by the user.

                Do NOT make up a context if none is provided!

                Return ONLY a JSON object with:
                - idiom: the expression the user asks about
                - context: the situation in which they heard it

                If no context is provided, use the JSON value null, NOT the string "null".
                If no idiom is identifiable, use JSON value null, NOT the string "null".
                Do not change anything about the user's context.

                Do not explain the idiom.
                NEVER infer a context from the idiom's meaning.

                Example: 
                {"idiom":"break the ice", "context":"My teacher said it yesterday."}

                Example:
                {"idiom": "hit the nail on the head", "context": null}
              `
            },
            {
              role: "user",
              content: context.lastResult?.[0]?.utterance ?? "",
            },
          ],
        }),

        // saving current idiom and user context
        onDone: [
          {
            guard: ({ event }) => !!event.output.idiom?.trim(), // going to Retrieval only if idiom was extracted

            actions: assign({
              currentIdiom: ({ event }) => event.output.idiom ?? "", // resetting/overwriting !!!!
              userContext: ({ context, event }) => { // resetting/overwriting !!!!
                // accounting for cases when the LLM might return "null" instead of null
                
                // prioritizing previously saved context (from Greeting; in case context came before idiom) !!!
                
                const value = event.output.context;

                if (
                  value == null ||
                  typeof value !== "string" ||
                  value.trim().toLowerCase() === "null"
                ) {
                  return context.userContext;
                }

                return context.userContext || value.trim();
              },

              // resetting upon successful extraction of new idiom
              generatedExample: "",
              hasExplainedIdiom: false,
            }),

            target: "CheckContext", // no retrieval here
          },
          {
            target: "Clarify",
          },
        ],

        onError: {
          actions: ({ event }) => {
            console.error("Extraction failed:", event.error);
          },
          target: "Error",
        },
      },
    }, // ExtractRequest end
    CheckIdiom: {
      // checking if the user has provided an idiom (and an idiom has been retrieved)


      always: [

        // no idiom provided
        {
          guard: ({ context }) =>
            !context.currentIdiom.trim(),

          target: "AskIdiom", // asking for idiom
        },

        // user wants an example
        {
          guard: ({ context }) =>
            context.interpretation?.topIntent === "request_example" ||
            context.interpretation?.topIntent === "different_example",

          target: "GiveContext", // sending to select example
        },
        {
          guard: ({ context }) =>
            context.interpretation?.topIntent === "more_details",

          target: "GiveDetails",
        },

        // else: continue on default pipeline
        {
          target: "CheckContext",
        },
      ],
    }, // CheckIdiom end
    AskIdiom: {
      // the user has not yet provided an idiom, so the system asks for it

      entry: {
        type: "spst.speak",
        params: {
          utterance: "Which idiom would you like to know about?",
        },
      },
      on: {
        SPEAK_COMPLETE: "GetIdiom",
      },
    }, // AskIdiom end
    GetIdiom: {
      // user provides idiom later in the dialogue

      entry: [
        assign({
          lastResult: null,
          interpretation: null,
        }),
        { type: "spst.listen" },
      ],

      on: {
        RECOGNISED: {
          actions: [
            assign({
              interpretation: ({ event }) => event.nluValue ?? null }),

            assign(({ context, event }) => ({
              lastResult: event.value,
              messages: [
                ...context.messages,
                {
                  role: "user" as const,
                  content: event.value[0].utterance,
                },
              ],
              noInputCount: 0,
            })),
          ],
        },

        LISTEN_COMPLETE: [
          {
            guard: ({ context }) => !!context.lastResult &&
              context.interpretation?.topIntent === "new_idiom" &&
              getIntentConfidence(context.interpretation, "new_idiom") > 0.65,
            target: "ExtractRequest",
          },
          {
            target: "Clarify", // sending back to beginning basically
          },
        ],
      },
    }, // GetIdiom end
    CheckContext: {
      // checking if the user has already provided context
      always: [
        {
          guard: ({ context }) => context.userContext.trim().length > 0,

          // setting dialogue intent
          actions: assign({
            dialogueIntent: "explain_context"
          }),

          target: "GetCompletion", // sending to LLM
        },
        {
          target: "AskContext", // if not: ask for context
        },
      ],
    }, // CheckContext end
    AskContext: {
      // user has not provided context, so system asks for context
      // before general explanation !!!

      entry: {
        type: "spst.speak",
        params: {
          utterance: "Where did you hear this expression, or how was it used?",
        },
      },
      on: {
        SPEAK_COMPLETE: "GetContext",
      },
    }, // AskContext end
    GetContext: {
      // the user provides context for the idiom
      // store context
      // give to LLM

      initial: "Listen",

      on: {
        // deciding next dialogue step
        LISTEN_COMPLETE: [

          // user provides context
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "user_context" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "user_context") > 0.65,

            // storing user context
            actions: assign({
              userContext: ({ context }) =>
                context.lastResult?.[0]?.utterance ?? "",
              // setting dialogue intent
              dialogueIntent: "explain_context",
            }),

            target: "GetCompletion" // sending to LLM
          },

          // user does not have context
          {
            guard: ({ context }) =>
              !!context.lastResult &&
              context.interpretation?.topIntent === "no_context" &&
              getIntentConfidence(context.interpretation, "no_context") > 0.65,
            target: "ConfirmExample",
          },

          // user wants the system to provide a context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "request_example" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "request_example") > 0.65,
            target: "GiveContext"
          },

          // else:
          {
            guard: ({ context }) => !!context.lastResult,
            target: "Clarify"
          },

          // third consecutive time no input
          {
            guard: ({ context }) =>
              !context.lastResult && context.noInputCount >= 3,
            target: ".NoInputFinal",
          },

          // first or second no input
          {
            guard: ({ context }) => !context.lastResult,
            target: ".NoInput",
          },
        ],
      },

      states: {
        Listen: {
          entry: [
            assign({
              lastResult: null,
              interpretation: null,
            }),
            { type: "spst.listen" },
          ],

          on: {
            // saving utterance contents
            RECOGNISED: {
              actions: [
                assign({interpretation: ({ event }) => event.nluValue ?? null}), // extracting intent
              
                assign(({ context, event }) => ({
                    lastResult: event.value, // updating lastResult with what user said
                    messages: [
                      ...context.messages, // appending to convo history
                      {
                        role: "user" as const, // as user message
                        content: event.value[0].utterance,
                      },
                    ],
                    noInputCount: 0, // resetting no input counter if utterance recognised !!!
                })),
              ],
            },

            // or handling no input
            ASR_NOINPUT: {
              // handling no input counter
              actions: assign(({ context }) => ({
                    lastResult: null, // no Input -> clear lastResult
                    interpretation: null, // clear interpretation
                    noInputCount: context.noInputCount + 1, // increasing the count by one
              })),
            }, // ASR_NOINPUT end
          },
        },
        NoInput: { // handling first and second time no input
          entry: {
            type: "spst.speak",
            params: ({ context }) => ({
                utterance:
                  context.noInputCount === 1
                    ? `I didn't hear anything. Please try again.` // if count === 1
                    : `I still didn't hear anything. Please try one more time`, // if count === 2 (ALSO IF 0 !!!)
            }),
          },
          on: { SPEAK_COMPLETE: "Listen" },
        }, // NoInput end        
        NoInputFinal: { // handling third time no input
          entry: {
            type: "spst.speak",
            params: { utterance: `I didn't hear anything. Click to try again! ` }, 
          },
          on: { SPEAK_COMPLETE: "#DM.WaitToStart" }, // sending the user back to the beginning after three times of no input
        }, // NoInputFinal end
      }, // states end  
    }, // GetContext end
    ConfirmExample: {
      // system lets the user know it will provide an example

      entry: {
        type: "spst.speak",
        params: {
          utterance: "No worries, I'll give you an example.",
        },
      },
      on: {
        SPEAK_COMPLETE: "GiveContext",
      },
    }, // ConfirmExample end
    GiveContext: {
      // the user wants to hear an example context from the system
      // or: the user wants to hear another example context from the system
      // directly to LLM (will generate)

      entry:

          // setting dialogue intent
          assign({
            dialogueIntent: "request_example",
          }),

      always: {
        target: "GetCompletion",
      },
    }, // GiveContext end
    GiveDetails: {
      // the user wants to know more details about the idiom (register, difficulty, why it's idiomatic)

      // sending straight to generation step
      always: {
        target: "GetCompletion",
      },

    }, // GiveDetails end
    GetCompletion: {
      // the LLM is prompted with the retrieved context
      invoke: {
        src: "fetchLLM",

        // making retrieved data for current idiom usable for conditional prompting
        input: ({ context }) => ({
          messages: [
            {
              role: "system" as const,

              // conditional system prompt
              content: `
                You are an English idiom tutor for language learners.
                You give short, concise answers.

                CURRENT IDIOM:
                ${context.currentIdiom}

                CURRENT TASK:
                ${context.dialogueIntent}

                ${
                  context.dialogueIntent === "request_example"
                    ? `
                      Generate a natural example sentence using the current idiom.

                      Your task is to help an English language learner understand how the idiom is used in the generated example.

                      Explain:
                      - What the idiom means in this particular sentence.
                      - Why the speaker uses the expression in this situation.
                      - The tone or intention, if relevant.

                      IMPORTANT:
                      - Give one example sentence.
                      - Do not repeat previous examples from the conversation.
                      - Explain in natural, simple English.
                      - Give approximately 2 short sentences of explanation.
                      - Do not invent additional situations.

                      Output the example sentence followed by its explanation.
                    `

                    : context.dialogueIntent === "explain_context"
                    ? `
                      USER'S CONTEXT:
                      ${context.userContext}

                      Explain what the idiom means in this situation.
                      Explain the speaker's likely intention.
                      The speaker might not be the user.
                      Do not invent details about the situation.
                    `

                    : context.dialogueIntent === "more_details"
                    ? `
                      The learner wants additional information about the idiom "${context.currentIdiom}".

                      Explain:

                      1. Why the expression is idiomatic or figurative.

                      2. Its register and when it is appropriate to use.

                      3. Its difficulty and how hard it is for language learners to master.

                      4. Any additional relevant nuances of its meaning.

                      Do not simply repeat the general definition.
                      Do not invent origins, cultural facts or usage restrictions.
                      If you are unsure about something, say so.
                      Do not invent facts.

                      Explain in 3-4 clear sentences suitable for an English language learner.
                    `

                    : `
                      Explain the general meaning of the idiom.
                      Give a short, clear explanation.
                    `
                }

                Keep your response concise and suitable for spoken dialogue.
              `,
            },
            // giving the LLM the entire context (to avoid generating duplicate examples)
            ...context.messages,
          ],
        }),

      onDone: {
        actions: assign({
          messages: ({ context, event }) => [
            ...context.messages,
            {
              role: "assistant",
              content: event.output,
            },
          ],
        }),
        target: "SpeakResponse",
        },

        onError: {
          actions: ({ event }) => {
            console.error("Completion failed:", event.error);
          },
          target: "Error",
        },
      },
    }, // GetCompletion end
    SpeakResponse: {
      // the system speaks the LLM generated response

      entry: {
        type: "spst.speak",
        params: ({ context }) => ({
            utterance: context.messages[context.messages.length - 1].content,
        }),
      },
      on: {
        SPEAK_COMPLETE: {
          // system has provided explanation now
          actions: assign({
            hasExplainedIdiom: true,
          }),
          target: "AnythingMore",
        },
      },
    }, // SpeakResponse end
    AnythingMore: {
      // system asks if there's anything else the user wants to know

      entry: {
        type: "spst.speak",
        params: {
          utterance: "Anything more?"
        },
      },
      on: {
        SPEAK_COMPLETE: "Greeting.Ask",
      },
    }, // AnythingMore end
    Error: {
      // handling possible invocation failures

      entry: {
        type: "spst.speak",
        params: {
          utterance:
            "Sorry, something went wrong. Please try again.",
        },
      },

      on: {
        SPEAK_COMPLETE: "Greeting.Ask",
      },
    }, // Error end
    Done: {
      // end state

      entry: {
        type: "spst.speak",
        params: {
          utterance: "Thank you for exploring idioms with me! Click to ask again."
        },
      },
      on: {
        SPEAK_COMPLETE: "WaitToStart",
      },

    }, // Done end
  },
});

const dmActor = createActor(dmMachine, {}).start();

// logs
dmActor.subscribe((state) => {
  console.group("State update");

  console.log("State value:", state.value);
  console.log("State context:", state.context);
  console.log("Dialogue intent:", state.context.dialogueIntent);

  console.log("Detected intent:", state.context.interpretation?.topIntent);

  console.log("Current idiom:", state.context.currentIdiom);
  console.log("User context:", state.context.userContext);

  console.log("Selected example:", state.context.generatedExample);

  console.log("No-input count:", state.context.noInputCount); // checking the no input counter

  console.groupEnd();
});

// user interface

// changing button text
export function setupButton(element: HTMLButtonElement) {
  element.addEventListener("click", () => {
    dmActor.send({ type: "CLICK" });
  });

  dmActor.subscribe((snapshot) => {
    const meta: { view?: string } = Object.values(
      snapshot.context.spstRef.getSnapshot().getMeta(),
    )[0] || { view: undefined };

    // displaying "Click" during WaitToStart
    if (snapshot.matches("WaitToStart")) {
      element.textContent = "Click";
    } else {
      element.textContent = meta.view ?? "";
    }
  });
}

// updating dialogue context values
export function setupDialogueDisplay() {
  const idiomElement = document.querySelector<HTMLParagraphElement>(
    "#current-idiom"
  );

  const contextElement = document.querySelector<HTMLParagraphElement>(
    "#user-context"
  );

  const exampleElement = document.querySelector<HTMLParagraphElement>(
    "#generated-example"
  );

  dmActor.subscribe((snapshot) => {
    const {
      currentIdiom,
      userContext,
      dialogueIntent,
      messages,
    } = snapshot.context;

    const lastAssistantMessage = [...messages]
      .reverse()
      .find((message) => message.role === "assistant");

    if (idiomElement) {
      idiomElement.textContent =
        currentIdiom || "No idiom selected yet";
    }

    if (contextElement) {
      contextElement.textContent =
        userContext || "No context provided yet";
    }

    if (exampleElement) {
      exampleElement.textContent =
        dialogueIntent === "request_example"
          ? lastAssistantMessage?.content ?? "No example yet"
          : "No example selected yet";
    }
  });
}
