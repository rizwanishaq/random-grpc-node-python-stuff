const grpc = require("grpc");
const path = require("path");
Int64 = require("node-int64");
const protoLoader = require("@grpc/proto-loader");
const {
  int16ToFloat32,
  bufferToInt16,
  bufferToFloat32,
} = require("./utils/utils");

// package definition
const packageDefinition = protoLoader.loadSync(
  path.resolve(__dirname, "proto/grpc_service.proto"),
  {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  }
);

const protoDescriptor = grpc.loadPackageDefinition(packageDefinition);

// inference package
const identifierPackage = protoDescriptor.inference;

const client = new identifierPackage.GRPCInferenceService(
  "100.100.100.52:8001",
  grpc.credentials.createInsecure()
);

const inferenceRequest = ({ audio_data, sr_data }) => {
  return new Promise((resolve, reject) => {
    const absValue = audio_data.map((sample) => {
      return Math.abs(sample);
    });
    const maxValue = Math.max(...absValue);
    const normalized = audio_data.map((sample) => {
      return sample / maxValue;
    });

    const request = {
      model_name: "riva-asr",
      parameters: {
        sequence_id: { int64_param: parseInt(10000) },
        sequence_start: { bool_param: 1 },
        sequence_end: { bool_param: 0 },
      },
      inputs: [
        {
          name: "AUDIO_SIGNAL",
          datatype: "FP32",
          shape: [1, audio_data.length],
        },
        {
          name: "SAMPLE_RATE",
          datatype: "UINT32",
          shape: [1, 1],
        },
      ],

      outputs: [
        { name: "AUDIO_FEATURES" },
        { name: "AUDIO_PROCESSED" },
        { name: "voiced" },
        { name: "CHARACTER_PROBABILITIES" },
      ],
      raw_input_contents: [
        Buffer.from(normalized.buffer),
        Buffer.from(sr_data.buffer),
      ],
    };

    // console.log(request);

    client.ModelInfer(request, (err, response) => {
      if (err) {
        console.log(err);
        return reject(`Error during request -> ${err}`);
      }
      // console.log(bufferToFloat32(response["raw_output_contents"]));
      console.log(response["raw_output_contents"]);
      return resolve({
        response: response,
      });
    });
  });
};

module.exports = inferenceRequest;
