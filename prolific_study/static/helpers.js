function getUrlParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

function shuffle(array) {
  let arr = array.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}


const ALL_COLORS = [
  "red",
  "orange",
  "yellow",
  "green",
  "blue",
  "purple",
  "pink",
  "brown",
  "grey"
];

function circularDistance(a, b, wheel) {
  const n = wheel.length;
  const ia = wheel.indexOf(a);
  const ib = wheel.indexOf(b);

  if (ia === -1 || ib === -1) {
    throw new Error(`Color not in wheel: ${a}, ${b}`);
  }

  const d = Math.abs(ia - ib);
  return Math.min(d, n - d);
}

function sampleDistractors(correctColor, n = 2) {
  // Guard against non-wheel colors
  if (!correctColor || correctColor === "white") {
    return shuffle(
      ALL_COLORS.filter(c => c !== "black")
    ).slice(0, n);
  }

  const MIN_DIST = 3;

  const candidates = ALL_COLORS.filter(c =>
    c !== correctColor &&
    circularDistance(c, correctColor, ALL_COLORS) >= MIN_DIST
  );

  if (candidates.length < n) {
    throw new Error(`Not enough distractors for ${correctColor}`);
  }

  return shuffle(candidates).slice(0, n);
}



function saveData() {
    var completion_date = new Date();
    jsPsych.data.get().push({ success: true, completion_time: completion_date, exp_type: "gesture-beat-expt-pilot" });
  
    var full_data = jsPsych.data.get().json();
    console.log(full_data);
    $.ajax({
        type: "POST",
        url: "https://mlepori.pythonanywhere.com/save-json",
        data: JSON.stringify({ 'prolific_data': jsPsych.data.dataProperties, 'data': full_data, 'dir_path': "data/experiment_2_visualization_pilot" }),
        contentType: "application/json"
    })
        .done(function () {
            window.location.href = "finish.html";
        })
        .fail(function () {
            alert("A problem occurred while writing to the database. Please contact the researcher for more information.")
        })
  }

function showSeekWarning() {
    document.getElementById("seeking-alert").style.display = "block";
}

function disableSeeking(v) {
    // Disable seeking
    var supposedCurrentTime = 0;
    v.addEventListener('timeupdate', function() {
    if (!v.seeking) {
        supposedCurrentTime = v.currentTime;
    }
    });
    // prevent user from seeking
    v.addEventListener('seeking', function() {
        // guard agains infinite recursion:
        // user seeks, seeking is fired, currentTime is modified, seeking is fired, current time is modified, ....
        var delta = v.currentTime - supposedCurrentTime;
        if (Math.abs(delta) > 0.01) {
            console.log("Seeking is disabled");
            v.currentTime = supposedCurrentTime;
        }
        showSeekWarning();

    });
    // delete the following event handler if rewind is not required
    v.addEventListener('ended', function() {
    // reset state in order to allow for rewind
        supposedCurrentTime = 0;
    });
}