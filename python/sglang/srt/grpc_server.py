import grpc
from concurrent import futures
import time
import logging
import dataclasses
import asyncio

from google.protobuf import empty_pb2
from google.protobuf import struct_pb2

from sglang.srt.proto import server_pb2, server_pb2_grpc
from sglang.srt.server import (
    tokenizer_manager,
    GenerateReqInput,
    EmbeddingReqInput,
    UpdateWeightFromDiskReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
    GetWeightsByNameReqInput,
    OpenSessionReqInput,
    CloseSessionReqInput,
    generate_request,
    encode_request,
    classify_request,
    launch_engine,
    ServerArgs,
)

scheduler_info = {
    "scheduler_name": "default_scheduler",
    "scheduler_version": "1.0"
}
__version__ = "1.0.0"

class SGLangServiceServicer(server_pb2_grpc.SGLangServiceServicer):
    def Health(self, request, context):
        return empty_pb2.Empty()

    def HealthGenerate(self, request, context):
        return empty_pb2.Empty()

    def GetModelInfo(self, request, context):
        result = {
            "model_path": tokenizer_manager.model_path,
            "tokenizer_path": tokenizer_manager.server_args.tokenizer_path,
            "is_generation": tokenizer_manager.is_generation,
        }
        return struct_pb2.Struct(fields=result)

    def GetServerInfo(self, request, context):
        server_info = {
            **dataclasses.asdict(tokenizer_manager.server_args),
            **scheduler_info,
            "version": __version__,
        }
        return struct_pb2.Struct(fields=server_info)

    def FlushCache(self, request, context):
        tokenizer_manager.flush_cache()
        return empty_pb2.Empty()

    def StartProfile(self, request, context):
        tokenizer_manager.start_profile()
        return empty_pb2.Empty()

    def StopProfile(self, request, context):
        tokenizer_manager.stop_profile()
        return empty_pb2.Empty()

    def UpdateWeightsFromDisk(self, request, context):
        async def update_weights():
            obj = UpdateWeightFromDiskReqInput(path=request.path)
            success, message = await tokenizer_manager.update_weights_from_disk(obj, None)
            return server_pb2.UpdateWeightsResponse(success=success, message=message)
        return asyncio.run(update_weights())

    def InitWeightsUpdateGroup(self, request, context):
        async def init_weights():
            obj = InitWeightsUpdateGroupReqInput(
                master_address=request.master_address,
                master_port=request.master_port,
                rank_offset=request.rank_offset,
                world_size=request.world_size,
                group_name=request.group_name,
                backend=request.backend,
            )
            success, message = await tokenizer_manager.init_weights_update_group(obj, None)
            return server_pb2.UpdateWeightsResponse(success=success, message=message)
        return asyncio.run(init_weights())

    def UpdateWeightsFromDistributed(self, request, context):
        async def update_weights():
            obj = UpdateWeightsFromDistributedReqInput(
                name=request.name,
                dtype=request.dtype,
                shape=request.shape,
            )
            success, message = await tokenizer_manager.update_weights_from_distributed(obj, None)
            return server_pb2.UpdateWeightsResponse(success=success, message=message)
        return asyncio.run(update_weights())

    def GetWeightsByName(self, request, context):
        async def get_weights():
            obj = GetWeightsByNameReqInput(name=request.name, truncate_size=request.truncate_size)
            try:
                result = await tokenizer_manager.get_weights_by_name(obj, None)
                return struct_pb2.Struct(fields=result)
            except Exception as e:
                raise RuntimeError(f"Failed to get weights by name: {e}")
        return asyncio.run(get_weights())

    def OpenSession(self, request, context):
        async def open_session():
            obj = OpenSessionReqInput(session_id=request.session_id)
            try:
                result = await tokenizer_manager.open_session(obj, None)
                return server_pb2.SessionResponse(session_id=result)
            except Exception as e:
                raise RuntimeError(f"Failed to open session: {e}")
        return asyncio.run(open_session())

    def CloseSession(self, request, context):
        async def close_session():
            obj = CloseSessionReqInput(session_id=request.session_id)
            try:
                await tokenizer_manager.close_session(obj, None)
                return empty_pb2.Empty()
            except Exception as e:
                raise RuntimeError(f"Failed to close session: {e}")
        return asyncio.run(close_session())

    def Generate(self, request, context):
        async def generate():
            obj = GenerateReqInput(
                text=request.text,
                sampling_params=request.sampling_params,
                rid=request.rid,
                return_logprob=request.return_logprob,
                logprob_start_len=request.logprob_start_len,
                top_logprobs_num=request.top_logprobs_num,
                return_text_in_logprobs=request.return_text_in_logprobs,
            )
            try:
                result = await generate_request(obj, request)
                return server_pb2.GenerateResponse(result=result)
            except Exception as e:
                raise RuntimeError(f"Failed to generate: {e} req: {request}")
        return asyncio.run(generate())

    def Encode(self, request, context):
        async def encode():
            obj = EmbeddingReqInput(text=request.text)
            try:
                result = await encode_request(obj, None)
                return server_pb2.EmbeddingResponse(result=result)
            except ValueError as e:
                raise RuntimeError(f"Failed to encode: {e}")
        return asyncio.run(encode())

    def Classify(self, request, context):
        async def classify():
            obj = EmbeddingReqInput(text=request.text)
            try:
                result = await classify_request(obj, None)
                return server_pb2.EmbeddingResponse(result=result)
            except ValueError as e:
                raise RuntimeError(f"Failed to classify: {e}")
        return asyncio.run(classify())


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_pb2_grpc.add_SGLangServiceServicer_to_server(SGLangServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server_args = ServerArgs(model_path='/home/jupyter/LLMASR/model')
    launch_engine(server_args=server_args)
    serve()