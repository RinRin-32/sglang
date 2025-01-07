import grpc
import json
import time
from typing import List
from sglang.srt.proto import completion_pb2, completion_pb2_grpc


def load_input_embeddings(file_path: str) -> List[List[List[float]]]:
    """
    Load embeddings from a file.
    Each line in the file should be a JSON-encoded list of lists of floats.
    """
    embeddings_list = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                embeddings = json.loads(line.strip())
                # Ensure embeddings are in the correct format (list of lists of floats)
                if isinstance(embeddings, list) and all(isinstance(i, list) for i in embeddings):
                    embeddings_list.append(embeddings)
                else:
                    print(f"Invalid embeddings format in line: {line}")
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}. Error: {e}")
    return embeddings_list


def create_completion_request(embeds: List[List[List[float]]], max_tokens: int = 100) -> completion_pb2.CompletionRequest:
    """
    Create a gRPC CompletionRequest with input embeddings.
    """
    # Convert the embeds to the correct message format
    embeds_3d = [completion_pb2.EmbedsList(embeds=sublist) for sublist in embeds]

    return completion_pb2.CompletionRequest(
        input_embeds=completion_pb2.InputEmbeds(embeds_3d=embeds_3d),  # Send the nested list of floats as ListofEmbeds
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stream=True,
        stop=[],
        ignore_eos=False,
    )


def main():
    # Server address
    server_address = "localhost:60001"
    embed_file_path = "/home/jupyter/LLMASR/test_embeds.txt" #path to preprocessed embed file

    # Load embeddings from the file
    embeddings_list = load_input_embeddings(embed_file_path)

    # Establish a connection to the gRPC server
    with grpc.insecure_channel(server_address) as channel:
        stub = completion_pb2_grpc.CompletionServiceStub(channel)

        # Iterate through each embedding and send requests
        for idx, embeds in enumerate(embeddings_list):
            try:
                print(f"Sending embeddings from line {idx + 1}")

                request = create_completion_request(embeds=embeds)
                # Start the timer
                start_time = time.time()
                response_iterator = stub.Complete(request)

                print("Server response:")
                for response in response_iterator:
                    print(f"Text: {response.text}, Finished: {response.finished}")
                    if response.finished:
                        break

                # Stop the timer
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Time taken for embeddings from line {idx + 1}: {elapsed_time:.2f} seconds")

            except grpc.RpcError as e:
                print(f"gRPC error: {e.code()} - {e.details()}")


if __name__ == "__main__":
    main()