import grpc
import ml_service_pb2
import ml_service_pb2_grpc


def run():
    # Connect to the server
    channel = grpc.insecure_channel('localhost:50051')
    stub = ml_service_pb2_grpc.MLServiceStub(channel)

    # Load data
    response = stub.LoadData(ml_service_pb2.LoadDataRequest(dataset_path="path/to/dataset"))
    print("LoadData Response:", response.message)

    # Train model
    response = stub.TrainModel(ml_service_pb2.TrainModelRequest(epochs=10, learning_rate=0.001))
    print("TrainModel Response:", response.message, "Accuracy:", response.training_accuracy)


if __name__ == "__main__":
    run()