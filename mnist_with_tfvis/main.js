// ***********************************************
// Handwritten Digit Classification in the Browser
// ***********************************************

// code adapted from: https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/

import {MnistData} from './data.js';

// ***
// *** Our "Main" Code that Runs on Page Initialization ***
// ***
document.addEventListener('DOMContentLoaded', run);
async function run() {  
  // load and display the data
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  // // model creation and training
  // const model = getModel();
  // tfvis.show.modelSummary({name: 'Model Architecture'}, model);
  // console.log('Training Model...');
  // await train(model, data);
  // console.log('Model Training Complete.');

  // // model evaluation
  // await showAccuracy(model, data);
  // await showConfusion(model, data);

}



// ***
// *** 1. Define the Model ***
// ***

function getModel() {
  console.log('Generating Model...');
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;  
  
  // Add first convolutional layer. Remember to specify inputShape. 


  // Add a MaxPooling layer to downsample using max values.  
  

  // Repeat another conv2d + maxPooling stack. 

  
  // Flatten the output of the 2D filters into a 1D vector for feeding to dense layer


  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).


  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model



  console.log('Model Generated');
  return model;
}


// ***
// *** 2. Train the Model ***
// ***

async function train(model, data) {
  // parameters for monitoring model training
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', styles: { height: '1000px' }
  };
  // this gets passed into the fit function to monitor training metrics
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
  // define training parameteres
  const NUM_EPOCHS = 10;
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  // prepare input tensors
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  // execute training
  return // define model fit function here
}


// ***
// *** 3. Evaluate the Model ***
// ***
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  // argMax() on tensor returns the indices of the max value along the specified axis
  const labels = testData.labels.argMax([-1]);
  const preds = model.predict(testxs).argMax([-1]);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(
      container, {values: confusionMatrix}, classNames);

  labels.dispose();
}

// ***
// *** Display Data Samples in a TFVIS Visor ***
// ***

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}