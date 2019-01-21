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
            [Column("PredeictedLabel")]
            public string PredictedLabel;
        }

        static void Main(string[] args)
        {
            // Instantizing
            var mlContext = new MLContext();

            
        }
    }
}
