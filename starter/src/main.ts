
import "./style.css";
import { setupButton, setupDialogueDisplay } from "./dm.ts";

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <div class="app-container">

    <header class="welcome">
      <div class="welcome-icon">📖</div>

      <h1>
        Idiomagic<br />
        <span>What do they actually mean?</span>
      </h1>

      <p class="subtitle">
        Your personal English idiom explainer
      </p>
    </header>

    <main class="dashboard">

      <section class="info-card">
        <h2>Current idiom</h2>
        <p id="current-idiom">
          No idiom selected yet
        </p>
      </section>

      <section class="info-card">
        <h2>User context</h2>
        <p id="user-context">
          No context provided yet
        </p>
      </section>

      <section class="info-card">
        <h2>Selected example</h2>
        <p id="selected-example">
          No example selected yet
        </p>
      </section>

    </main>

    <div class="card">
      <button id="counter" type="button">Click</button>
    </div>

  </div>
`;

// Keep your existing SpeechState button functionality
setupButton(
  document.querySelector<HTMLButtonElement>("#counter")!
);

// Display dialogue variables and update them automatically
setupDialogueDisplay();
