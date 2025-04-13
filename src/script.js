import {
    HandLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from "/src/knear.js";

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const drawUtils = new DrawingUtils(canvasCtx);
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;

let rightAnswers = 0;

let correctFrames = 0;

const clockFace = document.querySelector('.clockFace');

let coolingDown = false;

const machine = new kNear(4)

let image = document.querySelector("#myimage");

let data = null;
let questions = null;
let currentQuestion = null;

fetch('./public/handData.json')
    .then(response => response.json())
    .then(json => {
        data = json;
        console.log("Ingeladen JSON-data:", data);
        loadKNNModel(data)
    })
    .catch(error => console.error("Fout bij laden van JSON:", error));

fetch('./public/vragen.json')
    .then(response => response.json())
    .then(json => {
        questions = json;
        console.log("Ingeladen JSON-data:", questions);
        getRandomQuestion()
    })
    .catch(error => console.error("Fout bij laden van JSON:", error));


function getRandomQuestion() {
    const index = Math.floor(Math.random() * questions.length);
    currentQuestion = questions[index];
    document.getElementById("questionBox").textContent = currentQuestion.question;
}

function onUserPointsTo(hour) {
    if (coolingDown) return;

    if (currentQuestion && String(hour) === String(currentQuestion.answer)) {
        correctFrames++
        if (correctFrames >= 30) {

            clockFace.classList.add('green');

            setTimeout(() => {
                clockFace.classList.remove('green');
            }, 2000);

            coolingDown = true;

            setTimeout(() => {
                coolingDown = false;
            }, 2000);

            getRandomQuestion();
            correctFrames = 0;
            rightAnswers++
            document.getElementById("resultsBox").textContent = `Aantal oefeningen goed: ${rightAnswers}`

        }


    } else {
        correctFrames = 0;
    }
}

function loadKNNModel(data) {
    data.pointUp.forEach(handData => machine.learn(handData, 'pointUp'));
    data.pointDown.forEach(handData => machine.learn(handData, 'pointDown'));
    data.pointLeft.forEach(handData => machine.learn(handData, 'pointLeft'));
    data.pointRight.forEach(handData => machine.learn(handData, 'pointRight'));
    console.log("KNN model geladen en getraind.");
}


/********************************************************************
 // CREATE THE POSE DETECTOR
 ********************************************************************/
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
    console.log("model loaded, you can start webcam");

    enableWebcamButton.addEventListener("click", (e) => enableCam(e));
};

/********************************************************************
 // START DE WEBCAM
 ********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            // document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
 // START DE VOORSPELLINGEN
 ********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now());

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.landmarks.length > 0) {
        let hand = results.landmarks[0];
        // drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {color: "#00FF00", lineWidth: 5});
        // drawUtils.drawLandmarks(hand, {radius: 4, color: "#FF0000", lineWidth: 2});

        let handData = hand.map((o) => [o.x, o.y, o.z]).flat();

        let prediction = machine.classify(handData);

        console.log(prediction)


        let hour = null;
        if (prediction === 'pointUp') hour = 12;
        if (prediction === 'pointRight') hour = 3;
        if (prediction === 'pointDown') hour = 6;
        if (prediction === 'pointLeft') hour = 9;


        if (hour !== null) {
            onUserPointsTo(hour);
        }
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

/********************************************************************
 // LOG HAND COORDINATES IN DE CONSOLE
 ********************************************************************/
function logAllHands() {
    for (let hand of results.landmarks) {
        console.log(hand[4]);
    }
}

/********************************************************************
 // START DE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createHandLandmarker();
}
