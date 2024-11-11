import grpc
from concurrent import futures
import time
import ml_service_pb2
import ml_service_pb2_grpc
import os
import zipfile
import pytorch_lightning as pl


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    def LoadData(self, request, context):
        # Save the uploaded .zip file to a local directory
        file_path = f"./uploads/{request.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the received bytes to a .zip file
        with open(file_path, "wb") as f:
            f.write(request.file_data)

        # Extract the .zip file to a folder
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                extract_path = "./uploads/extracted_data"
                zip_ref.extractall(extract_path)
                message = f"Data extracted to {extract_path}"
                success = True
        except zipfile.BadZipFile:
            message = "Failed to load data: Invalid ZIP file."
            success = False

        return ml_service_pb2.LoadDataResponse(success=success, message=message)

    def TrainModel(self, request, context):
        print(f"Training model with {request.epochs} epochs and learning rate {request.learning_rate}")
        training_accuracy = 0.85
        success = True

        # TODO:
        # считать данныеб подать в classifier

        model = LightningPerceptronClassifier(
            data_root=root / "data" / "MNIST_DATA",
            input_dim=28 * 28,
            hidden_dim=hyperparameters.hidden_dim,
            output_dim=10,
            learning_rate=hyperparameters.learning_rate,
            batch_size=hyperparameters.batch_size
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=artifacts_dir,
            filename=model_filename,
            save_top_k=1,
            monitor="Validation loss"
        )
        trainer = pl.Trainer(
            max_epochs=hyperparameters.epochs,
            default_root_dir=artifacts_dir,
            callbacks=[checkpoint_callback]
        )
        trainer.fit(model)

        return ml_service_pb2.TrainModelResponse(success=success, message="Training completed",
                                                 training_accuracy=training_accuracy)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
