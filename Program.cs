using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace c_sharp_machine_learning
{
    class Program
    {
        /// <summary>
        /// IrisData is the model that shows the structure of the data schema.
        /// </summary>
        public class IrisData
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Label;
        }

        /// <summary>
        /// IrisPrediction will return the predicted result.
        /// </summary>
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }


        static void Main(string[] args)
        {
            // Instantizing ML.NET env
            var mlContext = new MLContext();

            // read the data from iris-data.txt and store it in the training var
            var reader = mlContext.Data.CreateTextReader<IrisData>(separatorChar: ',', hasHeader: true);
            IDataView trainingDataView = reader.Read("iris-data.txt");

            // Transform your data 
            // Assign numeric values to text in the label column, because only numbers can be processed during model training.
            // Add a learning algorithm to the PipeLine
            // Convert the label back into a text
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
               .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
               .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features"))
               .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            // train the model based on the data set provided
            var model = pipeline.Fit(trainingDataView);

            // Use model to predect label of Iris
            var prediction = model.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(
                new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f,
                });


            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.ReadLine();
        }
    }
}
