import grpc
from sglang.srt.proto import server_pb2, server_pb2_grpc

def run():
    # Connect to the gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = server_pb2_grpc.SGLangServiceStub(channel)

    # Create a GenerateReqInput message
    request = server_pb2.GenerateReqInput(
        text="Hello, this is a test."
    )

    # Call the Generate method
    try:
        response = stub.Generate(request)
        print("Generate response:", response.result)
    except grpc.RpcError as e:
        print(f"RPC failed: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run()