import traceback
from typing import Any, AsyncGenerator, Callable, Dict, List

import grpc

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
            # Handle input based on whether it's prompt or input_embeds
            if request.HasField("prompt"):
                input_text = request.prompt
                input_embeds = None
            elif request.HasField("input_embeds"):
                input_text = None
                input_embeds = self._parse_input_embeds(request.input_embeds)
            else:
                raise ValueError("Neither 'prompt' nor 'input_embeds' provided.")

            # Prepare the adapted request
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

            # Process request through tokenizer manager
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
            yield final_response

        except Exception as e:
            error_msg = f"Error in gRPC Complete: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            await context.abort(grpc.StatusCode.INTERNAL, error_msg)

    def _parse_input_embeds(self, input_embeds: completion_pb2.InputEmbeds) -> List[List[float]]:
        """
        Parse input embeddings from the gRPC request into a usable format.
        """
        if input_embeds.embeds_3d:
            # Process the 3D embeddings correctly
            return [list(chunk.embeds) for chunk in input_embeds.embeds_3d]
        elif input_embeds.embeds_2d:
            # Process the 2D embeddings as a flat list
            return [input_embeds.embeds_2d]
        else:
            raise ValueError("Invalid or empty input_embeds provided.")