from src.pipeline.training_pipeline import TraingingPipeline

if __name__ == "__main__":
    # Create an instance of the TrainingPipeline class
    pipeline = TraingingPipeline()
    
    # Run the training pipeline
    pipeline.run_pipeline()