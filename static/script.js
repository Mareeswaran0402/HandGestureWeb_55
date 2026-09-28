import {
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.js";


// =========================================================
// HTML ELEMENTS
// =========================================================

const video =
    document.getElementById("video");

const canvas =
    document.getElementById("overlay");

const ctx =
    canvas.getContext("2d");


const gestureElement =
    document.getElementById("gesture");

const confidenceElement =
    document.getElementById("confidence");

const holdElement =
    document.getElementById("holdMetric");

const holdBar =
    document.getElementById("holdBar");

const confidenceBar =
    document.getElementById("confidenceBar");

const sentenceElement =
    document.getElementById("sentence");

const statusElement =
    document.getElementById("status");

const topStatus =
    document.getElementById("topStatus");

const cameraMessage =
    document.getElementById("cameraMessage");


// =========================================================
// SETTINGS
// =========================================================

// 0.3 second gesture hold
const HOLD_TIME = 300;

// Send landmarks to Render approximately every 120 ms
const SERVER_INTERVAL = 120;


// =========================================================
// VARIABLES
// =========================================================

let handLandmarker = null;

let stream = null;

let running = false;

let lastServerRequest = 0;

let serverBusy = false;


let currentGesture = "NO HAND";

let currentConfidence = 0;


let previousGesture = "";

let gestureStartTime = 0;

let gestureAlreadyAdded = false;


let sentence = [];


// =========================================================
// INITIALIZE MEDIAPIPE
// =========================================================

async function initializeMediaPipe() {

    statusElement.textContent =
        "Loading hand detector...";


    // Load MediaPipe WebAssembly files
    const vision =
        await FilesetResolver.forVisionTasks(

            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"

        );


    // Create hand detector
    handLandmarker =
        await HandLandmarker.createFromOptions(

            vision,

            {

                baseOptions: {

                    modelAssetPath:
                        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",

                    // Use GPU if available
                    delegate: "GPU"

                },


                runningMode: "VIDEO",


                numHands: 1,


                minHandDetectionConfidence:
                    0.5,


                minHandPresenceConfidence:
                    0.5,


                minTrackingConfidence:
                    0.5

            }

        );


    statusElement.textContent =
        "Hand detector ready";

}


// =========================================================
// START CAMERA
// =========================================================

async function startCamera() {

    try {

        stream =
            await navigator.mediaDevices
                .getUserMedia({

                    video: {

                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        },

                        facingMode: "user"

                    },

                    audio: false

                });


        video.srcObject =
            stream;


        await video.play();


        running = true;


        cameraMessage.style.display =
            "none";


        topStatus.textContent =
            "LIVE";


        statusElement.textContent =
            "Show your hand";


        requestAnimationFrame(
            cameraLoop
        );


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );


        statusElement.textContent =
            "Camera permission denied";

    }

}


// =========================================================
// CAMERA LOOP
// =========================================================

function cameraLoop(timestamp) {

    if (!running) {
        return;
    }


    // Make overlay the same size as camera
    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;


    // -----------------------------------------------------
    // RUN MEDIAPIPE LOCALLY
    // -----------------------------------------------------

    if (handLandmarker) {

        const result =
            handLandmarker.detectForVideo(

                video,

                performance.now()

            );


        // =================================================
        // HAND FOUND
        // =================================================

        if (

            result.handLandmarks &&

            result.handLandmarks.length > 0

        ) {

            const hand =
                result.handLandmarks[0];


            // ---------------------------------------------
            // SAVE 21 LANDMARKS
            // ---------------------------------------------

            const landmarks =
                hand.map(

                    point => ({

                        x: point.x,

                        y: point.y,

                        z: point.z

                    })

                );


            // ---------------------------------------------
            // DRAW IMMEDIATELY
            // ---------------------------------------------

            drawLandmarks(
                landmarks
            );


            // ---------------------------------------------
            // SEND ONLY LANDMARKS TO RENDER
            // ---------------------------------------------

            if (

                timestamp
                - lastServerRequest
                >= SERVER_INTERVAL

                &&

                !serverBusy

            ) {

                lastServerRequest =
                    timestamp;


                predictOnServer(
                    landmarks
                );

            }


        } else {

            // ---------------------------------------------
            // NO HAND
            // ---------------------------------------------

            drawLandmarks([]);

            resetRecognition();

        }

    }


    // Continue every animation frame
    requestAnimationFrame(
        cameraLoop
    );

}


// =========================================================
// SEND LANDMARKS TO RENDER
// =========================================================

async function predictOnServer(
    landmarks
) {

    serverBusy = true;


    try {

        const response =
            await fetch(

                "/predict",

                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body: JSON.stringify({

                        landmarks:
                            landmarks

                    })

                }

            );


        if (!response.ok) {

            throw new Error(
                `HTTP ${response.status}`
            );

        }


        const data =
            await response.json();


        if (!data.success) {

            throw new Error(
                data.error
            );

        }


        // ---------------------------------------------
        // SERVER RESULT
        // ---------------------------------------------

        currentGesture =
            data.gesture;


        currentConfidence =
            Number(
                data.confidence || 0
            );


        updatePredictionUI();

        handleHold();


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        statusElement.textContent =
            "Prediction server delay";

    }


    finally {

        serverBusy = false;

    }

}


// =========================================================
// DRAW HAND LANDMARKS
// =========================================================

function drawLandmarks(
    landmarks
) {

    ctx.clearRect(

        0,

        0,

        canvas.width,

        canvas.height

    );


    if (
        !landmarks ||
        landmarks.length !== 21
    ) {

        return;

    }


    // -----------------------------------------------------
    // CONNECTIONS
    // -----------------------------------------------------

    const connections = [

        [0,1],
        [1,2],
        [2,3],
        [3,4],

        [0,5],
        [5,6],
        [6,7],
        [7,8],

        [0,9],
        [9,10],
        [10,11],
        [11,12],

        [0,13],
        [13,14],
        [14,15],
        [15,16],

        [0,17],
        [17,18],
        [18,19],
        [19,20],

        [5,9],
        [9,13],
        [13,17]

    ];


    // -----------------------------------------------------
    // GREEN LINES
    // -----------------------------------------------------

    ctx.strokeStyle =
        "lime";

    ctx.lineWidth =
        4;

    ctx.lineCap =
        "round";


    for (
        const [start, end]
        of connections
    ) {

        ctx.beginPath();


        ctx.moveTo(

            landmarks[start].x
            * canvas.width,

            landmarks[start].y
            * canvas.height

        );


        ctx.lineTo(

            landmarks[end].x
            * canvas.width,

            landmarks[end].y
            * canvas.height

        );


        ctx.stroke();

    }


    // -----------------------------------------------------
    // GREEN POINTS
    // -----------------------------------------------------

    ctx.fillStyle =
        "lime";


    for (
        const point
        of landmarks
    ) {

        ctx.beginPath();


        ctx.arc(

            point.x
            * canvas.width,

            point.y
            * canvas.height,

            6,

            0,

            Math.PI * 2

        );


        ctx.fill();

    }

}


// =========================================================
// UPDATE PREDICTION UI
// =========================================================

function updatePredictionUI() {

    gestureElement.textContent =
        currentGesture;


    confidenceElement.textContent =
        currentConfidence.toFixed(1)
        + "%";


    confidenceBar.style.width =
        Math.min(
            currentConfidence,
            100
        )
        + "%";


    if (
        currentGesture ===
        "UNCERTAIN"
    ) {

        gestureElement.style.color =
            "#f4ca5d";


        topStatus.textContent =
            "UNCERTAIN";

    }

    else if (
        currentGesture ===
        "NO HAND"
    ) {

        gestureElement.style.color =
            "#ef6670";


        topStatus.textContent =
            "READY";

    }

    else {

        gestureElement.style.color =
            "#35e27f";


        topStatus.textContent =
            "DETECTING";

    }

}


// =========================================================
// 0.3 SECOND HOLD
// =========================================================

function handleHold() {

    // -----------------------------------------------------
    // UNCERTAIN
    // -----------------------------------------------------

    if (
        currentGesture ===
        "UNCERTAIN"
    ) {

        previousGesture = "";

        gestureStartTime = 0;

        gestureAlreadyAdded =
            false;

        setHold(0);

        return;

    }


    // -----------------------------------------------------
    // NEW GESTURE
    // -----------------------------------------------------

    if (
        currentGesture !==
        previousGesture
    ) {

        previousGesture =
            currentGesture;


        gestureStartTime =
            Date.now();


        gestureAlreadyAdded =
            false;

    }


    // -----------------------------------------------------
    // CALCULATE HOLD
    // -----------------------------------------------------

    const held =
        Date.now()
        - gestureStartTime;


    const progress =
        Math.min(

            100,

            (
                held
                / HOLD_TIME
            ) * 100

        );


    setHold(
        progress
    );


    // -----------------------------------------------------
    // ADD WORD
    // -----------------------------------------------------

    if (

        held >= HOLD_TIME

        &&

        !gestureAlreadyAdded

    ) {

        sentence.push(
            currentGesture
        );


        gestureAlreadyAdded =
            true;


        updateSentence();


        statusElement.textContent =
            "Added: "
            + currentGesture;

    }

}


// =========================================================
// HOLD UI
// =========================================================

function setHold(
    value
) {

    value =
        Math.max(
            0,
            Math.min(100, value)
        );


    holdElement.textContent =
        value.toFixed(0)
        + "%";


    holdBar.style.width =
        value
        + "%";

}


// =========================================================
// RESET
// =========================================================

function resetRecognition() {

    currentGesture =
        "NO HAND";


    currentConfidence =
        0;


    previousGesture =
        "";


    gestureStartTime =
        0;


    gestureAlreadyAdded =
        false;


    updatePredictionUI();


    setHold(0);

}


// =========================================================
// SENTENCE
// =========================================================

function getSentenceText() {

    let text = "";


    for (
        const item of sentence
    ) {

        if (
            item === "|"
        ) {

            text += " ";

        }

        else {

            if (

                text.length > 0

                &&

                !text.endsWith(" ")

            ) {

                text += " ";

            }


            text += item;

        }

    }


    return text.trim();

}


function updateSentence() {

    sentenceElement.textContent =
        getSentenceText();

}


// =========================================================
// ADD WORD
// =========================================================

function addCurrentWord() {

    if (

        currentGesture ===
        "NO HAND"

        ||

        currentGesture ===
        "UNCERTAIN"

    ) {

        return;

    }


    sentence.push(
        currentGesture
    );


    updateSentence();

}


// =========================================================
// SPACE
// =========================================================

function addSpace() {

    if (

        sentence.length

        &&

        sentence[
            sentence.length - 1
        ] !== "|"

    ) {

        sentence.push("|");

        updateSentence();

    }

}


// =========================================================
// DELETE
// =========================================================

function deleteLast() {

    if (
        sentence.length
    ) {

        sentence.pop();

        updateSentence();

    }

}


// =========================================================
// CLEAR
// =========================================================

function clearSentence() {

    sentence = [];

    updateSentence();

}


// =========================================================
// SPEAK
// =========================================================

function speakSentence() {

    const text =
        getSentenceText();


    if (!text) {

        statusElement.textContent =
            "Nothing to speak";

        return;

    }


    if (
        !("speechSynthesis" in window)
    ) {

        statusElement.textContent =
            "Speech not supported";

        return;

    }


    window.speechSynthesis.cancel();


    const utterance =
        new SpeechSynthesisUtterance(
            text
        );


    utterance.rate =
        1.0;


    utterance.pitch =
        1.0;


    utterance.volume =
        1.0;


    utterance.onstart =
        function () {

            statusElement.textContent =
                "Speaking...";

            topStatus.textContent =
                "SPEAKING";

        };


    utterance.onend =
        function () {

            statusElement.textContent =
                "Ready";

            topStatus.textContent =
                "LIVE";

        };


    utterance.onerror =
        function () {

            statusElement.textContent =
                "Speech error";

        };


    window.speechSynthesis.speak(
        utterance
    );

}


// =========================================================
// BUTTONS
// =========================================================

document.getElementById(
    "addWordBtn"
).addEventListener(
    "click",
    addCurrentWord
);


document.getElementById(
    "spaceBtn"
).addEventListener(
    "click",
    addSpace
);


document.getElementById(
    "speakBtn"
).addEventListener(
    "click",
    speakSentence
);


document.getElementById(
    "deleteBtn"
).addEventListener(
    "click",
    deleteLast
);


document.getElementById(
    "clearBtn"
).addEventListener(
    "click",
    clearSentence
);


document.getElementById(
    "finishBtn"
).addEventListener(
    "click",
    speakSentence
);


// =========================================================
// START EVERYTHING
// =========================================================

async function start() {

    try {

        await initializeMediaPipe();

        await startCamera();

    }

    catch (error) {

        console.error(
            error
        );


        statusElement.textContent =
            "Could not load hand detector";

    }

}


start();