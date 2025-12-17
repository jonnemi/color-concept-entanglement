/**************************************************************************
 * INITIALIZATION
 **************************************************************************/

var jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: false
});

// ==========================
// Capture Prolific info
// ==========================

let subject_id = jsPsych.data.getURLVariable("PROLIFIC_PID");

if (!subject_id) {
  subject_id = "DEBUG_LOCAL_USER";
}

const study_id   = jsPsych.data.getURLVariable("STUDY_ID");
const session_id = jsPsych.data.getURLVariable("SESSION_ID");
const experiment_type = jsPsych.data.getURLVariable("exp");

jsPsych.data.addProperties({
  subject_id: subject_id,
  study_id: study_id,
  session_id: session_id,
  experiment_type: experiment_type
});

// ==========================
// RNG seed (JS side only for logging)
// ==========================

const js_seed = jsPsych.randomization.setSeed();
jsPsych.data.addProperties({ js_rng_seed: js_seed });

/**************************************************************************
 * GLOBAL CONFIG
 **************************************************************************/

const DEBUG = false;
const MAX_DISTRACTOR_RATE = 0.10;

let exp = experiment_type;

if (!exp && DEBUG) {
  exp = "1"; // default to Experiment 1 locally
}

if (!["1", "2", "3", "4"].includes(exp)) {
  alert("Invalid experiment configuration.");
  throw new Error("Invalid experiment type");
}


/**************************************************************************
 * FETCH STIMULI FROM SERVER
 **************************************************************************/

async function fetchStimuli() {
  const params = new URLSearchParams({
    PROLIFIC_PID: subject_id
  });

  const response = await fetch(`/get_stimuli?${params.toString()}`);
  if (!response.ok) {
    throw new Error("Failed to fetch stimuli from server");
  }
  return await response.json();
}

/**************************************************************************
 * HELPER: color judgment trial
 **************************************************************************/

function colorJudgmentTrial(stim, n_trials) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="text-align:center">
        <img src="img/dataset/${stim.image_path}" style="max-width:400px;"><br><br>
        <b>What color is the ${stim.object} in the image?</b>
      </div>
    `,
    choices: shuffle([
      stim.correct_color,
      "white",
      ...sampleDistractors(stim.correct_color)
    ]),
    data: {
      task_type: "color_judgment",
      object: stim.object,
      percent_colored: stim.percent_colored,
      correct_color: stim.correct_color
    },
    on_finish: function (data) {
      data.is_distractor =
        ![stim.correct_color, "white"].includes(data.response);

      const cur = jsPsych.getProgressBarCompleted();
      jsPsych.setProgressBar(cur + (1 / n_trials));
    }
  };
}

/**************************************************************************
 * BUILD TIMELINE
 **************************************************************************/

var timeline = [];

function buildTimeline(stimuli) {

  if (DEBUG) {
    stimuli = jsPsych.randomization.sampleWithoutReplacement(stimuli, 5);
  }

  const n_trials = stimuli.length;

  stimuli = shuffle(stimuli);

  // ==========================
  // Instructions
  // ==========================

  timeline.push({
    type: jsPsychInstructions,
    pages: [
      `
      <div class="jspsych-content" style="width:900px;text-align:left;">
        <h2>Welcome!</h2>
        <p>
          In this study, you will see a series of images and answer a simple
          question about each one.
        </p>
        <p>
          Your task is to decide what color an object is based on the image.
          Please answer as accurately as possible.
        </p>
        <p>
          The study will take approximately <b>10 minutes</b>.
        </p>
        <p>
          Click <b>Next</b> to begin.
        </p>
      </div>
      `
    ],
    show_clickable_nav: true,
    allow_backward: false,
    on_finish: function (data) {
      data.task_type = "instructions";
    }
  });

  // ==========================
  // Main trials
  // ==========================

  stimuli.forEach(stim => {

    // ---- Experiment 2: numeric introspection ----
    if (experiment_type === "2") {
      timeline.push({
        type: jsPsychHtmlSliderResponse,
        stimulus: `
          <p>
            For any object, <b>x%</b> of its pixels should be colored
            for it to be considered that color.
          </p>
          <p>
            What value should <b>x%</b> be?
          </p>
        `,
        min: 0,
        max: 100,
        step: 1,
        labels: ["0%", "100%"],
        data: {
          task_type: "threshold_introspection",
          object: stim.object,
          percent_colored: stim.percent_colored
        }
      });
    }

    // ---- Experiment 4: free-form introspection ----
    if (experiment_type === "4") {
      timeline.push({
        type: jsPsychSurveyText,
        questions: [{
          prompt: `
            If you were attempting to determine the color of an object,
            what rule would you use to do this?
            <br><br>
            (20–50 words)
          `,
          rows: 6
        }],
        data: {
          task_type: "rule_introspection",
          object: stim.object
        },
        on_finish: function (data) {
          const text = data.response.Q0 || "";
          const wc = text.trim().split(/\s+/).length;
          data.word_count = wc;
          data.valid_response = wc >= 20 && wc <= 50;
        }
      });
    }

    // ---- Color judgment (all experiments) ----
    timeline.push(colorJudgmentTrial(stim, n_trials));
  });

  // ==========================
  // Sanity check & booting
  // ==========================

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: function () {
      const total = jsPsych.data.get()
        .filter({ task_type: "color_judgment" })
        .count();

      const distractors = jsPsych.data.get()
        .filter({ is_distractor: true })
        .count();

      const frac = distractors / total;

      jsPsych.data.addProperties({
        distractor_fraction: frac,
        excluded: frac >= MAX_DISTRACTOR_RATE
      });

      if (frac >= MAX_DISTRACTOR_RATE) {
        return `
          <b>Attention check failed.</b><br><br>
          You selected implausible colors too frequently.
        `;
      } else {
        return `<b>Thank you for participating!</b>`;
      }
    },
    choices: ["Continue"],
    on_finish: function (data) {
      data.task_type = "quality_check";
    }
  });

  // ==========================
  // Save data
  // ==========================

  timeline.push({
    type: jsPsychCallFunction,
    func: function () {
      fetch("/save-json", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          experiment_type: experiment_type,
          data: jsPsych.data.get().values()
        })
      });
    }
  });

  // ==========================
  // Finish
  // ==========================

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: "Click below to complete the study.",
    choices: ["Finish"],
    on_finish: function () {
      window.location.href = "finish.html";
    }
  });
}

/**************************************************************************
 * START EXPERIMENT
 **************************************************************************/

async function run_experiment() {
  try {
    const payload = await fetchStimuli();

    jsPsych.data.addProperties({
      sampling_seed: payload.seed
    });

    buildTimeline(payload.stimuli);
    jsPsych.run(timeline);

  } catch (err) {
    alert("Error loading experiment. Please contact the researcher.");
    console.error(err);
  }
}
