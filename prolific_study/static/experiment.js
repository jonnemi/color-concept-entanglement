/**************************************************************************
 * INITIALIZATION
 **************************************************************************/

var jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: false,
});

// ---------------------------------------------------------------------
// GLOBAL STATE
// ---------------------------------------------------------------------

let experimentEnded = false;
const MAX_DURATION_MS = 1000 * 30; // 30s test mode (use 30 * 60 * 1000 in prod)

/**************************************************************************
 * Capture Prolific info
 **************************************************************************/

let subject_id = jsPsych.data.getURLVariable("PROLIFIC_PID");
if (!subject_id) subject_id = "DEBUG_LOCAL_USER";

const study_id = jsPsych.data.getURLVariable("STUDY_ID");
const session_id = jsPsych.data.getURLVariable("SESSION_ID");

jsPsych.data.addProperties({
  subject_id,
  study_id,
  session_id,
});

// ---------------------------------------------------------------------
// Timeline container
// ---------------------------------------------------------------------

let timeline = [];

/**************************************************************************
 * FETCH PROFILE FROM SERVER
 **************************************************************************/

async function fetchProfile() {
  const params = new URLSearchParams({ PROLIFIC_PID: subject_id });
  const response = await fetch(`/get_profile?${params.toString()}`);
  if (!response.ok) throw new Error("Failed to fetch profile");
  return await response.json();
}

/**************************************************************************
 * SAFE TERMINATION
 **************************************************************************/

function safeEndExperiment(message, reason) {
  if (experimentEnded) return;
  experimentEnded = true;

  jsPsych.data.addProperties({
    exit_reason: reason,
    exit_time: Date.now(),
  });

  jsPsych.endExperiment(
    message + "<br><br>Please return the study on Prolific."
  );
}

/**************************************************************************
 * RENDERERS
 **************************************************************************/

function renderColorJudgment(q) {
  const choices = shuffle([
    q.target_color,
    "white",
    ...sampleDistractors(q.target_color, 2),
  ]);

  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="text-align:center">
        <img src="img/dataset/${q.image_path}" style="max-width:400px;"><br><br>
        <b>What color is the ${q.object} in the image?</b>
      </div>
    `,
    choices,

    data: {
      task_type: "color_judgment",
      object: q.object,
      stimulus_type: q.stimulus_type,
      percent_colored: q.percent_colored,
      variant_region: q.variant_region,
      target_color: q.target_color,
    },

    on_finish: function (data) {
      const chosen_label = choices[data.response];

      const allowed_answers =
        q.variant_region === "BG"
          ? ["white"]
          : [q.target_color, "white"];

      const is_wrong = !allowed_answers.includes(chosen_label);

      data.response_label = chosen_label;
      data.is_distractor = is_wrong;

      if (is_wrong) {
        safeEndExperiment(
          "You selected an implausible color.",
          "failed_distractor"
        );
        return;
      }

      const cur = jsPsych.getProgressBarCompleted();
      jsPsych.setProgressBar(cur + 1 / 106);
    },
  };
}


function renderSanity(q) {
  // TEXT sanity
  if (q.response_type === "text") {
    return {
      type: jsPsychSurveyText,
      questions: [{ prompt: q.prompt, rows: 3 }],
      data: {
        task_type: "sanity",
        sanity_id: q.sanity_id,
        correct_response: q.correct_response,
      },
      on_finish: function (data) {
        const response = (data.response.Q0 || "").trim().toLowerCase();
        const correct = q.correct_response.toLowerCase();

        data.response_label = response;
        data.passed = response === correct;

        if (!data.passed) {
          safeEndExperiment(
            "You did not pass the attention check.",
            "failed_attention"
          );
        }
      },
    };
  }

  // LIKERT sanity
  return {
    type: jsPsychSurveyLikert,
    questions: [
      {
        prompt: q.prompt,
        labels: q.options,
        required: true,
      },
    ],
    data: {
      task_type: "sanity",
      sanity_id: q.sanity_id,
      correct_response: q.correct_response,
    },
    on_finish: function (data) {
      const selectedIndex = data.response.Q0;
      const selectedLabel = q.options[selectedIndex];

      data.response_index = selectedIndex;
      data.response_label = selectedLabel;
      data.passed = selectedLabel === q.correct_response;

      if (!data.passed) {
        safeEndExperiment(
          "You did not pass the attention check.",
          "failed_attention"
        );
      }
    },
  };
}


function renderIntrospection(q) {
  return {
    type: jsPsychHtmlSliderResponse,
    stimulus: `
      <p>
        For any object, <b>x%</b> of its pixels should be colored
        for it to be considered that color.
      </p>
      <p>What value should <b>x%</b> be?</p>
    `,
    min: q.min,
    max: q.max,
    step: 1,
    labels: ["0%", "100%"],
    data: { task_type: "introspection" },
  };
}

/**************************************************************************
 * GLOBAL TIMEOUT
 **************************************************************************/

function startGlobalTimeout() {
  const start = Date.now();

  jsPsych.data.addProperties({
    experiment_start_time: start,
  });

  window.setTimeout(() => {
    if (experimentEnded) return;

    safeEndExperiment(
      "The study timed out.",
      "timeout"
    );
  }, MAX_DURATION_MS);
}

/**************************************************************************
 * BUILD TIMELINE
 **************************************************************************/

function buildTimeline(questions) {
  timeline = [];

  // Instructions + timer start
  timeline.push({
    type: jsPsychInstructions,
    pages: [
      `
      <div class="jspsych-content" style="width:900px;text-align:left;">
        <h2>Welcome!</h2>

        <p>
          You will see a series of images and answer questions about their colors.
        </p>

        <p>
          Please answer <b>carefully and accurately</b>.
        </p>

        <p><b>Important:</b></p>
        <ul>
          <li>You will be removed if you select an implausible color.</li>
          <li>You will be removed if you fail an attention check.</li>
          <li>The study has a <b>30-minute time limit</b>.</li>
        </ul>

        <p>Click <b>Next</b> to begin.</p>
      </div>
      `,
    ],
    show_clickable_nav: true,
    allow_backward: false,
    on_finish: startGlobalTimeout,
  });

  // Questions
  questions.forEach((q) => {
    if (q.question_type === "sanity") {
      timeline.push(renderSanity(q));
    } else if (q.question_type === "introspection") {
      timeline.push(renderIntrospection(q));
    } else {
      timeline.push(renderColorJudgment(q));
    }
  });

  // Save results
  timeline.push({
    type: jsPsychCallFunction,
    func: async function () {
      jsPsych.data.addProperties({
        exit_reason: "completed",
      });

      await fetch("/save_results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          PROLIFIC_PID: subject_id,
          data: jsPsych.data.get().values(),
        }),
      });
    },
  });

  // Finish
  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: "Click below to complete the study.",
    choices: ["Finish"],
    on_finish: () => {
      window.location.href = "finish.html";
    },
  });
}

/**************************************************************************
 * START EXPERIMENT
 **************************************************************************/

async function run_experiment() {
  try {
    const payload = await fetchProfile();

    jsPsych.data.addProperties({
      profile_id: payload.profile_id,
      profile_index: payload.profile_index,
    });

    buildTimeline(payload.questions);
    jsPsych.run(timeline);

  } catch (err) {
    alert("Error loading experiment. Please contact the researcher.");
    console.error(err);
  }
}
