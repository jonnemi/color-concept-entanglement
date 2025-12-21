/**************************************************************************
 * INITIALIZATION
 **************************************************************************/

var jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: false,
});

// ---------------------------------------------------------------------
// Capture Prolific info
// ---------------------------------------------------------------------

let subject_id = jsPsych.data.getURLVariable("PROLIFIC_PID");
if (!subject_id) {
  subject_id = "DEBUG_LOCAL_USER";
}

const study_id = jsPsych.data.getURLVariable("STUDY_ID");
const session_id = jsPsych.data.getURLVariable("SESSION_ID");

jsPsych.data.addProperties({
  subject_id: subject_id,
  study_id: study_id,
  session_id: session_id,
});

// ---------------------------------------------------------------------
// Timeline container
// ---------------------------------------------------------------------

let timeline = [];

/**************************************************************************
 * FETCH PROFILE FROM SERVER
 **************************************************************************/

async function fetchProfile() {
  const params = new URLSearchParams({
    PROLIFIC_PID: subject_id,
  });

  const response = await fetch(`/get_profile?${params.toString()}`);
  if (!response.ok) {
    throw new Error("Failed to fetch profile from server");
  }
  return await response.json();
}

/**************************************************************************
 * RENDERERS
 **************************************************************************/

function renderColorJudgment(q) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="text-align:center">
        <img src="img/dataset/${q.image_path}" style="max-width:400px;"><br><br>
        <b>What color is the ${q.object} in the image?</b>
      </div>
    `,
    choices: shuffle([
      q.target_color,
      "white",
      ...sampleDistractors(q.target_color, 2),
    ]),
    data: {
      task_type: "color_judgment",
      object: q.object,
      stimulus_type: q.stimulus_type,
      condition: q.condition,
      percent_colored: q.percent_colored,
      variant_region: q.variant_region,
      target_color: q.target_color,
    },
    on_finish: function (data) {
      data.is_distractor =
        ![q.target_color, "white"].includes(data.response);

      const cur = jsPsych.getProgressBarCompleted();
      jsPsych.setProgressBar(cur + 1 / 106);
    },
  };
}

function renderSanity(q) {
  if (q.response_type === "text") {
    return {
      type: jsPsychSurveyText,
      questions: [
        {
          prompt: q.prompt,
          rows: 3,
        },
      ],
      data: {
        task_type: "sanity",
        sanity_id: q.sanity_id,
        target_response: q.target_response,
      },
    };
  }

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
      target_response: q.target_response,
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
      <p>
        What value should <b>x%</b> be?
      </p>
    `,
    min: q.min,
    max: q.max,
    step: 1,
    labels: ["0%", "100%"],
    data: {
      task_type: "introspection",
    },
  };
}

/**************************************************************************
 * BUILD TIMELINE FROM PROFILE
 **************************************************************************/

function buildTimeline(questions) {
  timeline = [];

  // --------------------------------------------------
  // Instructions
  // --------------------------------------------------

  timeline.push({
    type: jsPsychInstructions,
    pages: [
      `
      <div class="jspsych-content" style="width:900px;text-align:left;">
        <h2>Welcome!</h2>
        <p>
          You will see a series of images along with multiple choice
          questions about the colors you see in each image.
        </p>
        <p>
          Please read each question carefully and answer as accurately
          as possible.
        </p>
        <p>
          The study will take approximately <b>10 minutes</b>.
        </p>
        <p>
          Click <b>Next</b> to begin.
        </p>
      </div>
      `,
    ],
    show_clickable_nav: true,
    allow_backward: false,
    on_finish: function (data) {
      data.task_type = "instructions";
    },
  });

  // --------------------------------------------------
  // Main questions (already ordered in profile)
  // --------------------------------------------------

  questions.forEach((q) => {
    if (q.question_type === "sanity") {
      timeline.push(renderSanity(q));
    } else if (q.question_type === "introspection") {
      timeline.push(renderIntrospection(q));
    } else {
      timeline.push(renderColorJudgment(q));
    }
  });

  // --------------------------------------------------
  // Save results
  // --------------------------------------------------

  timeline.push({
    type: jsPsychCallFunction,
    func: async function () {
      const values = jsPsych.data.get().values();

      await fetch("/save_results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          PROLIFIC_PID: subject_id,
          profile_id: values.length > 0 ? values[0].profile_id : null,
          data: values,
        }),
      });
    },
  });


  // --------------------------------------------------
  // Finish
  // --------------------------------------------------

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: "Click below to complete the study.",
    choices: ["Finish"],
    on_finish: function () {
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
