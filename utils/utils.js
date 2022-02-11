// This is passed in an unsigned 16-bit integer array. It is converted to a 32-bit float array.
// The first startIndex items are skipped, and only 'length' number of items is converted.
// Take from https://stackoverflow.com/questions/35234551/javascript-converting-from-int16-to-float32
const int16ToFloat32 = (inputArray) => {
  const output = new Float32Array(inputArray.length);
  for (let i = 0; i < inputArray.length; i++) {
    const int = inputArray[i];
    // If the high bit is on, then it is a negative number, and actually counts backwards.
    const float = int >= 0x8000 ? -(0x10000 - int) / 0x8000 : int / 0x7fff;
    output[i] = float;
  }
  return output;
};

// convert the buffer to float32
const floatTo16Bit = (buffer) => {
  let l = buffer.length;
  let output = new Int16Array(l);
  while (l--) {
    const s = (output[l] = Math.min(1, buffer[l]));
    output[l] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
};

const bufferToInt16 = (buffer) => {
  return new Int16Array(buffer.length / 2).map((val, index) =>
    buffer.readInt16LE(index * 2)
  );
};

const CalculateSUM = (arr) => arr.reduce((acum, val) => acum + val);

const CalculateSUMABS = (arr) =>
  arr.reduce((acum, val) => acum + Math.abs(val));

const CalculateMAXABS = (arr) =>
  arr.reduce((a, b) => Math.max(Math.abs(a), Math.abs(b)));

const CalculateRMS = (arr) =>
  Math.sqrt(
    arr.map((val) => val * val).reduce((acum, val) => acum + val) / arr.length
  );

const CalculateSTD = (arr) => {
  const mean = arr.reduce((acum, val) => acum + val) / arr.length;
  return Math.sqrt(
    arr
      .map((val) => val - mean)
      .map((val) => val * val)
      .reduce((acum, val) => acum + val) / arr.length
  );
};

const CalculateSOFTMAX = (arr) => {
  const sum = CalculateSUM(arr);
  return arr.map((val) => val / sum);
};

const CalculateABS = (arr) => {
  return arr.map((val) => Math.abs(val));
};

const prettier = (obj) => {
  return JSON.stringify(obj, null, 1);
};
const bufferToFloat32 = (buffer) => {
  return new Float32Array(buffer.length / 4).map((val, index) =>
    buffer.readFloatLE(index * 4)
  );
};

module.exports = {
  int16ToFloat32,
  floatTo16Bit,
  bufferToInt16,
  CalculateSUM,
  CalculateSUMABS,
  CalculateMAXABS,
  CalculateRMS,
  CalculateSTD,
  CalculateABS,
  CalculateSOFTMAX,
  prettier,
  bufferToFloat32,
};
