const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';
const image = document.getElementById("exampleImage")
let movenet;

async function loadModel(){
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });
  
  const exampleTensor = tf.zeros([1, 192,192,3], "int32");
  const imageTensor = tf.browser.fromPixels(image);
  console.log(imageTensor.shape);
  
  
  const cropStartPoint = [15, 170, 0];
  const cropSize = [345, 345, 3];
  const croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);
  
  const resizedTensor = tf.image.resizeBilinear(croppedTensor, [192,192], true).toInt();
  console.log(resizedTensor.shape);

  const tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
  const arrayOutput = await tensorOutput.array();
  
  console.log(arrayOutput);
  
  
  tensorOutput.dispose();
  movenet.dispose();
  
}

loadModel();