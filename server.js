const grpc = require("grpc");
const path = require("path");
const protoLoader = require("@grpc/proto-loader");

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

const ModelInfer = (call, callback) => {
  console.log(call.request);
  console.log(call.request.parameters);
};

/*
 * - Setting up the grpc server */
const server = new grpc.Server();
server.addService(identifierPackage.GRPCInferenceService.service, {
  ModelInfer: ModelInfer,
});
/*
 * - Binding the server the specific host and port */
server.bind("100.100.100.52:8001", grpc.ServerCredentials.createInsecure());

/*
 * - Starting the grpc server */
server.start();
console.log("Server started - 100.100.100.52:8001");
