import { assign, createActor, fromPromise, setup } from "xstate";
import { Settings, speechstate } from "speechstate";
import { PROXY_KEY, NLU_KEY } from "./credentials";
import { DMContext, DMEvents, IdiomRecord, Message, NLUObject, IdiomExample, ExtractedRequest } from "./types"; // added Message
import OpenAI from "openai";
import { QdrantClient } from "@qdrant/js-client-rest"; // importing qdrant

const REGION = "northeurope";

const openai = new OpenAI({
  baseURL: "http://localhost:11434/v1/",
  apiKey: "ollama",
  dangerouslyAllowBrowser: true,
});

// embedding helper function: converting text into 384-dimensional vector
const embed = async (input: string) =>
  openai.embeddings
    .create({
      model: "qwen3-embedding",
      input,
      dimensions: 384,
    })
    .then((result) => result.data[0].embedding);

// creating Qdrant client
const qdrant = new QdrantClient({
  host: "localhost",
  port: 6333,
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
  deploymentName: "idiom_explainer",
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

    // retrieval step: input: user query -> embedded, output: relevant text from qdrant
    queryRAG: fromPromise<IdiomRecord[], { query: string }>(
      async ({ input }) => {
        const embedding = await embed(input.query);

        // querying idioms collection
        const result = await qdrant.query("idioms", {
          query: embedding,
          with_payload: true,
          limit: 3,
        });
      
        // converting returned payloads into array
        const retrievedIdioms: IdiomRecord[] = result.points
          .filter((point) => point.payload != null)
          .map((point) => {
            const payload = point.payload!;

            return {
              idiom: String(payload.idiom ?? ""),
              meaning: String(payload.meaning ?? ""),
              explanation: String(payload.explanation ?? ""),
              register: String(payload.register ?? ""),
              difficulty: String(payload.difficulty ?? ""),
              example: Array.isArray(payload.example)
                ? payload.example as unknown as IdiomExample[]
                : [],
            };
          });

        //console.log("Qdrant payload:", retrievedIdioms);

        return retrievedIdioms;

      },
    ),
  },
}).createMachine({
  context: ({ spawn }) => ({
    spstRef: spawn(speechstate, { input: settings }),

    // Adding interpretation
    interpretation: null as NLUObject | null,

    // storing ASR results for current turn -> resets before going back to listening
    lastResult: null,

    // storing idiom under discussion
    currentIdiom: "",

    // storing context provided by user
    userContext: "",

    // dialogue history across multiple turns
    messages: [
      {
        role: "system",
        content:
          "You are a helpful assistant for University of Gothenburg students.",
      },
    ] as Message[],
    retrievedIdioms: [] as IdiomRecord[], // to store converted Qdrant arrays for current query
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
      on: { 
        CLICK: {
          actions: assign({
            noInputCount: 0, // resetting no input counter at restart
          }),
          target: "Greeting",
        },
      },
    }, // WaitToStart end
    Greeting: {
      // greeting + initial user ask is analyzed
      initial: "Prompt",
      on: {
        LISTEN_COMPLETE: [
          
          // detecting intent + recognizing utterance
          // recognition successful -> RAG pipeline

          // user asks about new idiom
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "new_idiom" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "new_idiom") > 0.65,
            target: "ExtractRequest"
          },
          // user provides context
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "user_context" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "user_context") > 0.65,
            target: "GetContext" // ???
          },
          // user wants the system to provide a context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "request_example" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "request_example") > 0.65,
            target: "GiveContext" // ???
          },
          // user wants to hear different context example
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "different_example" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "different_example") > 0.65,
            target: "GiveContext" // different state ???
          },
          // user wants more in depth explanation of the current idiom
          {
            guard: ({ context }) => 
              !!context.lastResult &&
              context.interpretation?.topIntent === "more_details" &&
            // checking for confidence (should be over 65%)
              getIntentConfidence(context.interpretation, "more_details") > 0.65,
            target: "GiveDetails" // ???
          },

          // handling unclear input

          { target: "Clarify",
            guard: ({ context }) => !!context.lastResult, // CHECK !!!!
          },


          // handling no input

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
          entry: { type: "spst.speak", params: { utterance: `Welcome to: What do they actually mean? - your personal idiom explainer` } },
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
            "Sorry, I didn't understand. Could you rephrase your question?",
        },
      },
      on: {
        SPEAK_COMPLETE: "Greeting.Ask", // change to whatever state it came from !!!!
      },
    }, // Clarify end
    ExtractRequest: {
      // sending the user utterance to LLM to extract the new idiom and possibly the context
      invoke: {
        src: "extractRequest",
        input: ({ context }) => ({
          messages: [
            {
              role: "system" as const,
              content: `
                Extract the idiom and any situation or context provided by the user.

                Return ONLY a JSON object with:
                - idiom: the expression the user asks about
                - context: the situation in which they heard it

                If no context is provided, use null.
                If no idiom is identifiable, use null.
                Do not change anything about the user's context.

                Do not explain the idiom.

                Example: 
                {"idiom":"break the ice", "context":"My teacher said it yesterday."}
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
              userContext: ({ event }) => event.output.context ?? "", // resetting/overwriting !!!!
            }),
            target: "Retrieve", // ???
          },
          {
            target: "Clarify",
          },
        ],
      }, // CHECK !!!!
    }, // ExtractRequest end
    GetContext: {
      // the user has provided context for the idiom
      // store context
      // give to LLM
      // LLM uses retrieved RAG data (but does not retrieve again)
      // LLM speaks answer
    }, // GetContext end
    AskContext: {
      // user has not provided context, so system asks for context
      // before or after general explanation ???

      entry: {
        type: "spst.speak",
        params: {
          utterance: "Where did you hear this expression, or how was it used?",
        },
      },
      on: {
        SPEAK_COMPLETE: "Greeting", // change to AwaitContext ????
      },
    }, // AskContext end
    Retrieve: { 
      // invoking queryRAG to pass latest user utterance
      invoke: {
        src: "queryRAG",
        input: ({ context }) => ({
  		    query: context.currentIdiom, // querying using currentIdiom
        }),

        // keeping the latest two user utterances for retrieval
        /*
        input: ({ context }) => {
          // keeping only user messages
          const userMessages = context.messages.filter(
            (message) => message.role === "user",
          );

          // taking the two most recent user messages
          const recentUserMessages = userMessages.slice(-2);

          // combining them
          const query = recentUserMessages
            .map((message) => message.content)
            .join("\n");

          // control log
          console.log("RAG query:", query);

          return {
            query: query,
          };
        },
        */
        onDone: {
          actions: [
            
            // storing the retrieved info (for augmentation step)
            assign({
              retrievedIdioms: ({ event }) => event.output,
              //currentIdiom: ({ event }) => event.output[0]?.idiom ?? "", // !!!
            }),
          
            ({ event }) => {
              console.log("Retrieved idioms:", event.output);  // control log
            },
          ],

          target: "CheckContext",
        },
      },
    }, // Retrieve end
    CheckContext: {
      // checking if the user has already provided context
      always: [
        {
          guard: ({ context }) => context.userContext.trim().length > 0,
          target: "GetContext",
        },
        {
          target: "AskContext", // if not: ask for context
        },
      ],
    }, // CheckContext end
    GiveContext: {
      // the user wants to hear an example context from the system
      // possibly: the user wants to hear another example context from the system
    }, // GiveContext end
    GiveDetails: {
      // the user wants to know more details about the idiom (register, difficulty, why it's idiomatic)
    }, // GiveDetails end
    GetCompletion: {
      // the LLM is prompted with the retrieved context
      invoke: {
        src: "fetchLLM",
        input: ({ context }) => ({
          messages: [
            {
              role: "system" as const,

              // augmentation step: retrieved qdrant context -> system prompt
              content: `
                You are an assistant for language learners of English. You explain idioms.
                
                Use the retrieved information below to answer the user's question.
                If the answer is not supported by the retrieved information, 
                please let the user know that you do not have enough information to answer the question.

                RETRIEVED IDIOM INFORMATION START

                ${JSON.stringify(context.retrievedIdioms, null, 2)}

                RETRIEVED IDIOM INFORMATION END

                Explain the idiom in simple English.
                If the user provided context, explain what the idiom means in that context.

                The retrieved examples may contain both literal and idiomatic uses of the expression.
                Use the user's context (if available) to determine which interpretation is appropriate.

                When providing an example of idiomatic usage, prefer examples that clearly demonstrate the figurative meaning.
                Do not assume every retrieved example is idiomatic.

                Keep the responses concise and efficient.
              `,
            },
            ...context.messages
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
          target: "Greeting.Ask",
        },
      },
    }, // SpeakResponse end
  },
});

const dmActor = createActor(dmMachine, {}).start();

dmActor.subscribe((state) => {
  console.group("State update");
  console.log("State value:", state.value);
  console.log("State context:", state.context);
  console.log("No-input count:", state.context.noInputCount); // checking the no input counter
  console.groupEnd();
});

export function setupButton(element: HTMLButtonElement) {
  element.addEventListener("click", () => {
    dmActor.send({ type: "CLICK" });
  });
  dmActor.subscribe((snapshot) => {
    const meta: { view?: string } = Object.values(
      snapshot.context.spstRef.getSnapshot().getMeta(),
    )[0] || {
      view: undefined,
    };
    element.innerHTML = `${meta.view}`;
  });
}
