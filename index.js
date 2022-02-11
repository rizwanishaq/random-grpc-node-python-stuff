const main = async () => {
  const inferenceRequest = require("./requestSend.js");

  const audio_data = Float32Array.from(
    [...Array(1600)].map(() => Math.random())
  );
  // console.log(audio_data);

  let sr_data = new Uint32Array(1);
  sr_data[0] = 16000;
  // console.log(Buffer.from(sr_data.buffer));

  const response = await inferenceRequest({
    audio_data: audio_data,
    sr_data: sr_data,
  });
};

main();
