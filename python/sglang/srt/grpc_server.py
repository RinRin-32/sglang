import traceback
from typing import Any, AsyncGenerator, Callable, Dict, List
import grpc
from io import BytesIO
import numpy as np
import time

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.proto import completion_pb2, completion_pb2_grpc


class CompletionServicer(completion_pb2_grpc.CompletionServiceServicer):
    def __init__(
        self,
        generate_request: Callable[
            [GenerateReqInput], AsyncGenerator[Dict[str, Any], None]
        ],
    ):
        self.generate_request = generate_request

    async def Complete(
        self,
        request: completion_pb2.CompletionRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[completion_pb2.CompletionResponse, None]:
        try:
            start_time = time.time()
            input_text = None
            input_embeds = None

            # Handle input parsing
            parse_start = time.time()
            if request.HasField("prompt"):
                input_text = request.prompt
            elif request.HasField("input_embeds"):
                input_embeds = self._parse_input_embeds(request.input_embeds)
            elif request.HasField("embedding_file"):
                input_embeds = self._parse_embedding_file(request.embedding_file.file_content)
            else:
                raise ValueError("No valid input provided (prompt, input_embeds, or embedding_file).")
            parse_end = time.time()

            # Prepare the adapted request
            prepare_start = time.time()
            adapted_request = GenerateReqInput(
                text=input_text,
                input_embeds=input_embeds,
                sampling_params={
                    "max_new_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "min_p": request.min_p,
                    "frequency_penalty": request.frequency_penalty,
                    "presence_penalty": request.presence_penalty,
                    "stop": list(request.stop),
                    "ignore_eos": request.ignore_eos,
                },
                stream=request.stream,
            )
            prepare_end = time.time()

            # Process request through tokenizer manager
            processing_start = time.time()
            async for content in self.generate_request(adapted_request):
                response = completion_pb2.CompletionResponse(
                    text=content["text"],  # Send full text so far
                    finished=False,
                    usage=completion_pb2.Usage(
                        prompt_tokens=content["meta_info"]["prompt_tokens"],
                        completion_tokens=content["meta_info"]["completion_tokens"],
                    ),
                )
                yield response

            final_response = completion_pb2.CompletionResponse(
                text=content["text"],  # Final complete text
                finished=True,
                usage=completion_pb2.Usage(
                    prompt_tokens=content["meta_info"]["prompt_tokens"],
                    completion_tokens=content["meta_info"]["completion_tokens"],
                ),
            )
            processing_end = time.time()
            yield final_response

            # Total time logging
            end_time = time.time()
            print(f"=== Timing Report ===")
            print(f"Total Time: {end_time - start_time:.4f} seconds")
            print(f"Input Parsing Time: {parse_end - parse_start:.4f} seconds")
            print(f"Request Preparation Time: {prepare_end - prepare_start:.4f} seconds")
            print(f"Processing Time: {processing_end - processing_start:.4f} seconds")

        except Exception as e:
            error_msg = f"Error in gRPC Complete: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            await context.abort(grpc.StatusCode.INTERNAL, error_msg)

    def _parse_input_embeds(self, input_embeds: completion_pb2.InputEmbeds) -> List[List[float]]:
        """
        Parse input embeddings from the gRPC request into a usable format.
        """
        parse_start = time.time()
        try:
            if input_embeds.embeds_3d:
                # Process the 3D embeddings correctly
                result = [list(chunk.embeds) for chunk in input_embeds.embeds_3d]
            elif input_embeds.embeds_2d:
                # Process the 2D embeddings as a flat list
                result = [input_embeds.embeds_2d]
            else:
                raise ValueError("Invalid or empty input_embeds provided.")
            return result
        finally:
            parse_end = time.time()
            print(f"_parse_input_embeds Time: {parse_end - parse_start:.4f} seconds")

    def _parse_embedding_file(self, file_content: bytes) -> List[List[float]]:
        """
        Parse embeddings from a file sent as bytes.
        """
        parse_start = time.time()
        try:
            with BytesIO(file_content) as file_stream:
                # Assuming the file is saved as a NumPy array
                embeddings = np.load(file_stream, allow_pickle=True)
                if not isinstance(embeddings, np.ndarray):
                    raise ValueError("The embedding file does not contain a valid NumPy array.")
                return embeddings.tolist()
        except Exception as e:
            raise ValueError(f"Error parsing embedding file: {str(e)}")
        finally:
            parse_end = time.time()
            print(f"_parse_embedding_file Time: {parse_end - parse_start:.4f} seconds")